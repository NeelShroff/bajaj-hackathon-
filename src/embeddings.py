import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from openai import OpenAI, AsyncOpenAI
from langchain.schema import Document
from tiktoken import get_encoding
import asyncio
from pinecone import Pinecone, ServerlessSpec
import uuid
import json
import sys

from config import config

logger = logging.getLogger(__name__)

# Load tokenizer for token counting
TOKEN_ENCODING = get_encoding("cl100k_base")

class EmbeddingsManager:
    """Manages OpenAI embeddings and Pinecone index for document retrieval."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.async_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        
        try:
            self.pinecone = Pinecone(api_key=config.PINECONE_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            raise
            
        self.index = None
        self.documents = []
        self.dimension = 1536
        
        self.MAX_PAYLOAD_SIZE = 3.5 * 1024 * 1024
        self.MAX_VECTORS_PER_BATCH = 100
        
        self.initialize_index()

    def initialize_index(self):
        """Initializes the Pinecone index if it doesn't already exist."""
        index_name = config.PINECONE_INDEX_NAME
        
        try:
            existing_indexes = []
            for index in self.pinecone.list_indexes():
                existing_indexes.append(index.name)
            
            if index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index '{index_name}'...")
                self.pinecone.create_index(
                    name=index_name,
                    dimension=self.dimension,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region=config.PINECONE_ENVIRONMENT)
                )
                logger.info(f"Pinecone index '{index_name}' created.")
            
            self.index = self.pinecone.Index(index_name)
            logger.info(f"Connected to Pinecone index '{index_name}'.")
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {e}")
            raise

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

    async def _async_batch_create_embeddings(self, documents: List[Document], max_tokens_per_batch: int = 8191) -> Tuple[List[Tuple], List[Document]]:
        """
        Creates embeddings in dynamic batches, handling oversized chunks by splitting them.
        """
        all_embeddings_vectors = []
        processed_documents = []
        
        pre_processed_docs = []
        for doc in documents:
            text = doc.page_content
            text_tokens = len(TOKEN_ENCODING.encode(text))
            
            if text_tokens > max_tokens_per_batch:
                logger.warning(f"Chunk from doc '{doc.metadata.get('document_name', 'N/A')}' with {text_tokens} tokens exceeds API limit. Splitting.")
                sub_chunks = self._split_and_batch_text(text, max_tokens=max_tokens_per_batch)
                
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_chunk_doc = Document(
                        page_content=sub_chunk,
                        metadata={**doc.metadata, 'sub_chunk_index': i}
                    )
                    pre_processed_docs.append(sub_chunk_doc)
            else:
                pre_processed_docs.append(doc)

        batches_to_process = []
        current_batch_texts = []
        current_batch_docs = []
        current_batch_tokens = 0
        for doc in pre_processed_docs:
            text = doc.page_content
            text_tokens = len(TOKEN_ENCODING.encode(text))
            if current_batch_tokens + text_tokens <= max_tokens_per_batch:
                current_batch_texts.append(text)
                current_batch_docs.append(doc)
                current_batch_tokens += text_tokens
            else:
                batches_to_process.append({'texts': current_batch_texts, 'docs': current_batch_docs})
                current_batch_texts = [text]
                current_batch_docs = [doc]
                current_batch_tokens = text_tokens
        if current_batch_texts:
            batches_to_process.append({'texts': current_batch_texts, 'docs': current_batch_docs})

        async_tasks = [self._async_create_embeddings_api_call(batch['texts']) for batch in batches_to_process]
        results = await asyncio.gather(*async_tasks)

        pinecone_vectors = []
        processed_documents = []
        for batch_index, batch_result in enumerate(results):
            for i, embedding in enumerate(batch_result):
                doc = batches_to_process[batch_index]['docs'][i]
                metadata = {**doc.metadata, 'text': doc.page_content}
                pinecone_vectors.append((str(uuid.uuid4()), embedding, metadata))
                processed_documents.append(doc)

        return pinecone_vectors, processed_documents

    def _estimate_payload_size(self, vectors: List[Tuple]) -> int:
        """Estimate the JSON payload size for a batch of vectors."""
        if not vectors:
            return 0
        
        sample_vector = {
            'id': vectors[0][0],
            'values': vectors[0][1],
            'metadata': vectors[0][2]
        }
        sample_json = json.dumps(sample_vector)
        sample_size = len(sample_json.encode('utf-8'))
        
        estimated_size = len(vectors) * sample_size + 1000
        return estimated_size

    def _split_vectors_by_payload_size(self, vectors: List[Tuple]) -> List[List[Tuple]]:
        """Split vectors into batches that respect Pinecone's payload size limit."""
        batches = []
        current_batch = []
        
        for vector in vectors:
            test_batch = current_batch + [vector]
            
            if (len(test_batch) > self.MAX_VECTORS_PER_BATCH or 
                self._estimate_payload_size(test_batch) > self.MAX_PAYLOAD_SIZE):
                
                if current_batch:
                    batches.append(current_batch)
                    current_batch = [vector]
                else:
                    logger.warning(f"Vector {vector[0]} too large, truncating metadata")
                    truncated_vector = self._truncate_vector_metadata(vector)
                    batches.append([truncated_vector])
            else:
                current_batch = test_batch
        
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def _truncate_vector_metadata(self, vector: Tuple) -> Tuple:
        """Truncate vector metadata to fit within size limits."""
        vector_id, embedding, metadata = vector
        
        # Keep only essential metadata, now including the LLM-derived 'section_type'
        essential_metadata = {
            'document_name': metadata.get('document_name', ''),
            'section_id': metadata.get('section_id', ''),
            'section_type': metadata.get('section_type', ''), # NEW: Use the AI-derived section type
            'chunk_index': metadata.get('chunk_index', 0),
            'source': metadata.get('source', ''),
        }
        
        text = metadata.get('text', '')
        if len(text) > 1000:
            essential_metadata['text'] = text[:1000] + "..."
        else:
            essential_metadata['text'] = text
        
        return (vector_id, embedding, essential_metadata)

    def _split_and_batch_text(self, text: str, max_tokens: int = 8191) -> List[str]:
        """Splits a large string into smaller parts based on token count."""
        tokens = TOKEN_ENCODING.encode(text)
        sub_chunks = []
        for i in range(0, len(tokens), max_tokens):
            sub_chunk_tokens = tokens[i:i + max_tokens]
            sub_chunks.append(TOKEN_ENCODING.decode(sub_chunk_tokens))
        return sub_chunks
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text."""
        return self.client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=[text]
        ).data[0].embedding
    
    async def build_index_async(self, documents: List[Document]) -> None:
        """Build Pinecone index by upserting document vectors in properly sized batches."""
        if not documents:
            logger.warning("No documents provided for indexing")
            return
        
        logger.info(f"Building Pinecone index by upserting {len(documents)} documents.")
        
        vectors_to_upsert, self.documents = await self._async_batch_create_embeddings(documents)
        
        vector_batches = self._split_vectors_by_payload_size(vectors_to_upsert)
        
        logger.info(f"Upserting {len(vectors_to_upsert)} vectors in {len(vector_batches)} batches.")
        
        upsert_tasks = []
        for i, batch in enumerate(vector_batches):
            upsert_task = asyncio.to_thread(self.index.upsert, vectors=batch)
            upsert_tasks.append(upsert_task)
            
        await asyncio.gather(*upsert_tasks)
        
        logger.info(f"Pinecone index build completed.")
    
    def search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Search for similar documents using query embedding."""
        if self.index is None:
            raise ValueError("Pinecone index not initialized.")
        
        if k is None:
            k = config.TOP_K_RESULTS
        
        query_embedding = self.create_embedding(query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        search_results = []
        for match in results.matches:
            doc = Document(
                page_content=match.metadata.get('text', ''),
                metadata=match.metadata
            )
            search_results.append((doc, match.score))
        
        return search_results

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current Pinecone index."""
        if self.index is None:
            return {"status": "no_index"}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats['total_vector_count'],
                "dimension": stats['dimension'],
                "status": "ready"
            }
        except Exception as e:
            logger.error(f"Error getting Pinecone index stats: {e}")
            return {"status": "error"}