import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import faiss
from openai import OpenAI
from langchain.schema import Document

from config import config

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    """Manages OpenAI embeddings and FAISS index for document retrieval."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.index = None
        self.documents = []
        self.dimension = 1536  # OpenAI text-embedding-ada-002 dimension
        
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text."""
        embeddings = self.create_embeddings([text])
        return embeddings[0]
    
    def build_index(self, documents: List[Document]) -> None:
        """Build FAISS index from documents."""
        if not documents:
            logger.warning("No documents provided for indexing")
            return
        
        logger.info(f"Building FAISS index for {len(documents)} documents")
        
        # Extract texts from documents
        texts = [doc.page_content for doc in documents]
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.index.add(embeddings_array)
        
        # Store documents for retrieval
        self.documents = documents
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Search for similar documents using query embedding."""
        if self.index is None:
            raise ValueError("FAISS index not built. Call build_index() first.")
        
        if k is None:
            k = config.TOP_K_RESULTS
        
        # Create query embedding
        query_embedding = self.create_embedding(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Search
        scores, indices = self.index.search(query_vector, k)
        
        # Return documents with scores
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
        
        # Filter by threshold
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
        
        # Extract texts and create embeddings
        texts = [doc.page_content for doc in new_documents]
        embeddings = self.create_embeddings(texts)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Add to documents list
        self.documents.extend(new_documents)
        
        logger.info(f"Added {len(new_documents)} documents to index. Total: {self.index.ntotal}")
    
    def save_index(self, file_path: str = None) -> None:
        """Save FAISS index and documents to disk."""
        if file_path is None:
            file_path = config.FAISS_INDEX_PATH
        
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = f"{file_path}.faiss"
        faiss.write_index(self.index, index_path)
        
        # Save documents metadata
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
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load documents
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
    
    def batch_create_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Create embeddings in batches to handle rate limits."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.create_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        return all_embeddings 