import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import faiss
from openai import OpenAI, AsyncOpenAI
from langchain.schema import Document
from tiktoken import get_encoding
import asyncio

from config import config

logger = logging.getLogger(__name__)

# Load tokenizer for token counting
TOKEN_ENCODING = get_encoding("cl100k_base")

class EmbeddingsManager:
    """Manages OpenAI embeddings and FAISS index for document retrieval."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.async_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.index = None
        self.documents = []
        self.dimension = 1536
        
    async def _async_create_embeddings_api_call(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts using OpenAI API asynchronously."""
        try:
            response = await self.async_client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise

    async def _async_batch_create_embeddings(self, texts: List[str], max_tokens_per_batch: int = 8191) -> List[List[float]]:
        """
        Creates embeddings in parallel batches for maximum speed.
        """
        tasks = []
        current_batch_texts = []
        current_batch_tokens = 0
        
        for text in texts:
            text_tokens = len(TOKEN_ENCODING.encode(text))
            
            if text_tokens > max_tokens_per_batch:
                if current_batch_texts:
                    tasks.append(self._async_create_embeddings_api_call(current_batch_texts))
                logger.warning(f"Chunk with {text_tokens} tokens exceeds API limit. Processing as a single call.")
                tasks.append(self._async_create_embeddings_api_call([text]))
                current_batch_texts = []
                current_batch_tokens = 0
                continue

            if current_batch_tokens + text_tokens <= max_tokens_per_batch:
                current_batch_texts.append(text)
                current_batch_tokens += text_tokens
            else:
                tasks.append(self._async_create_embeddings_api_call(current_batch_texts))
                logger.info(f"Adding a new batch of {len(current_batch_texts)} chunks with {current_batch_tokens} tokens to the queue.")
                current_batch_texts = [text]
                current_batch_tokens = text_tokens
        
        if current_batch_texts:
            tasks.append(self._async_create_embeddings_api_call(current_batch_texts))
            logger.info(f"Adding final batch of {len(current_batch_texts)} chunks with {current_batch_tokens} tokens to the queue.")
            
        # Run all API calls in parallel
        results = await asyncio.gather(*tasks)
        
        all_embeddings = []
        for batch_result in results:
            all_embeddings.extend(batch_result)
        
        return all_embeddings

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text."""
        return self.client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=[text]
        ).data[0].embedding
    
    def build_index(self, documents: List[Document]) -> None:
        """Build FAISS index from documents."""
        if not documents:
            logger.warning("No documents provided for indexing")
            return
        
        logger.info(f"Building FAISS index for {len(documents)} documents")
        
        # Split oversized chunks first
        final_documents = []
        for doc in documents:
            text = doc.page_content
            text_tokens = len(TOKEN_ENCODING.encode(text))
            if text_tokens > 8191:
                sub_chunks = self._split_and_batch_text(text, max_tokens=8191)
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_chunk_doc = Document(
                        page_content=sub_chunk,
                        metadata={**doc.metadata, 'sub_chunk_index': i}
                    )
                    final_documents.append(sub_chunk_doc)
            else:
                final_documents.append(doc)
        
        self.documents = final_documents
        texts = [doc.page_content for doc in self.documents]

        # Use asyncio to run the asynchronous embedding function
        embeddings = asyncio.run(self._async_batch_create_embeddings(texts))
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_array)
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors from {len(self.documents)} chunks")
    
    def _split_and_batch_text(self, text: str, max_tokens: int = 8191) -> List[str]:
        """Splits a large string into smaller parts based on token count."""
        tokens = TOKEN_ENCODING.encode(text)
        sub_chunks = []
        for i in range(0, len(tokens), max_tokens):
            sub_chunk_tokens = tokens[i:i + max_tokens]
            sub_chunks.append(TOKEN_ENCODING.decode(sub_chunk_tokens))
        return sub_chunks

    def search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Search for similar documents using query embedding."""
        if self.index is None:
            raise ValueError("FAISS index not built. Call build_index() first.")
        
        if k is None:
            k = config.TOP_K_RESULTS
        
        query_embedding = self.create_embedding(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def search_with_threshold(self, query: str, threshold: float = None, k: int = None) -> List[Tuple[Document, float]]:
        """Search with similarity threshold filtering."""
        if threshold is None:
            threshold = config.SIMILARITY_THRESHOLD
        
        results = self.search(query, k)
        
        filtered_results = [
            (doc, score) for doc, score in results 
            if score >= threshold
        ]
        
        return filtered_results
    
    def add_documents(self, new_documents: List[Document]) -> None:
        """Add new documents to existing index."""
        if self.index is None:
            self.build_index(new_documents)
            return
        
        texts = [doc.page_content for doc in new_documents]
        embeddings = asyncio.run(self._async_batch_create_embeddings(texts))
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        self.index.add(embeddings_array)
        self.documents.extend(new_documents)
        
        logger.info(f"Added {len(new_documents)} documents to index. Total: {self.index.ntotal}")
    
    def save_index(self, file_path: str = None) -> None:
        """Save FAISS index and documents to disk."""
        if file_path is None:
            file_path = config.FAISS_INDEX_PATH
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        index_path = f"{file_path}.faiss"
        faiss.write_index(self.index, index_path)
        
        docs_path = f"{file_path}.pkl"
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Saved index to {index_path} and documents to {docs_path}")
    
    def load_index(self, file_path: str = None) -> bool:
        """Load FAISS index and documents from disk."""
        if file_path is None:
            file_path = config.FAISS_INDEX_PATH
        
        index_path = f"{file_path}.faiss"
        docs_path = f"{file_path}.pkl"
        
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            logger.warning(f"Index files not found: {file_path}")
            return False
        
        try:
            self.index = faiss.read_index(index_path)
            
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        if self.index is None:
            return {"status": "no_index"}
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "total_documents": len(self.documents),
            "index_type": type(self.index).__name__
        }
    
    def reset_index(self) -> None:
        """Reset the index and documents."""
        self.index = None
        self.documents = []
        logger.info("Index reset")