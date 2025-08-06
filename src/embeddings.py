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
from functools import lru_cache
import hashlib

from config import config

# Configure logging
logger = logging.getLogger(__name__)
TOKEN_ENCODING = get_encoding("cl100k_base")

class EmbeddingsManager:
    """
    Manages OpenAI embeddings and Pinecone index for document retrieval.
    Optimized for high-speed, parallel upsertion with cost efficiency.
    """
    
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
        
        # --- OPTIMIZATION: More aggressive batch sizes ---
        # Pinecone's serverless payload limit is ~4MB, so 3.5MB is a safe buffer
        self.MAX_PAYLOAD_SIZE = 3.5 * 1024 * 1024
        self.MAX_VECTORS_PER_BATCH = 5000
        self.MAX_CONCURRENT_UPSERTS = 10
        
        self.initialize_index()

    def initialize_index(self):
        """Initializes the Pinecone index if it doesn't already exist."""
        index_name = config.PINECONE_INDEX_NAME
        
        try:
            existing_indexes = [index.name for index in self.pinecone.list_indexes()]
            
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

    @lru_cache(maxsize=1000)
    def _cached_create_embedding(self, text_hash: str, text: str) -> List[float]:
        """Create embedding with caching for single synchronous queries."""
        try:
            response = self.client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
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

    async def _async_batch_create_embeddings(self, documents: List[Document], max_tokens_per_batch: int = 30000) -> Tuple[List[Tuple], List[Document]]:
        """
        Creates embeddings from a list of pre-chunked documents in dynamic batches.
        Optimized for concurrency and API token limits.
        """
        batches_to_process = []
        current_batch_texts = []
        current_batch_docs = []
        current_batch_tokens = 0
        for doc in documents:
            text = doc.page_content
            if not isinstance(text, str):
                logger.warning(f"Skipping document with non-string content: {type(text)}")
                continue

            text_tokens = len(TOKEN_ENCODING.encode(text))
            
            if current_batch_tokens + text_tokens <= max_tokens_per_batch and len(current_batch_texts) < 2048:
                current_batch_texts.append(text)
                current_batch_docs.append(doc)
                current_batch_tokens += text_tokens
            else:
                if current_batch_texts:
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
        
        estimated_size = len(vectors) * sample_size
        return estimated_size

    def _split_vectors_by_payload_size(self, vectors: List[Tuple]) -> List[List[Tuple]]:
        """
        OPTIMIZATION: Split vectors into batches that are intelligently sized
        to respect Pinecone's payload size limit.
        """
        batches = []
        current_batch = []
        
        for vector in vectors:
            test_batch = current_batch + [vector]
            
            # The key is to check the size and count *before* adding the next vector
            # This logic avoids creating a batch that's already too big
            if (len(test_batch) > self.MAX_VECTORS_PER_BATCH or 
                self._estimate_payload_size(test_batch) > self.MAX_PAYLOAD_SIZE):
                
                if current_batch:
                    batches.append(current_batch)
                    current_batch = [vector]
                else:
                    logger.warning(f"Vector {vector[0]} is too large for a single batch even with metadata truncation.")
                    # Fallback for extremely large single vectors
                    batches.append([self._truncate_vector_metadata(vector)])
            else:
                current_batch = test_batch
        
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def _truncate_vector_metadata(self, vector: Tuple) -> Tuple:
        """Truncate vector metadata to fit within size limits."""
        vector_id, embedding, metadata = vector
        
        essential_metadata = {
            'document_name': metadata.get('document_name', ''),
            'section_id': metadata.get('section_id', ''),
            'section_type': metadata.get('section_type', ''),
            'chunk_index': metadata.get('chunk_index', 0),
            'source': metadata.get('source', ''),
        }
        
        text = metadata.get('text', '')
        if len(text) > 1000:
            essential_metadata['text'] = text[:1000] + "..."
        else:
            essential_metadata['text'] = text
        
        return (vector_id, embedding, essential_metadata)

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text."""
        query_hash = hashlib.md5(text.encode()).hexdigest()
        return self._cached_create_embedding(query_hash, text)
    
    async def build_index_async(self, documents: List[Document]) -> None:
        """
        Build Pinecone index by upserting document vectors in parallel batches.
        Uses a semaphore to manage concurrency and avoid rate limits.
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return
        
        logger.info(f"Building Pinecone index by upserting {len(documents)} documents.")
        
        vectors_to_upsert, self.documents = await self._async_batch_create_embeddings(documents)
        
        vectors_to_upsert = [v for v in vectors_to_upsert if v[1] and isinstance(v[1], list) and len(v[1]) == self.dimension]
        
        vector_batches = self._split_vectors_by_payload_size(vectors_to_upsert)
        
        logger.info(f"Upserting {len(vectors_to_upsert)} vectors in {len(vector_batches)} batches.")

        # --- CORRECTED OPTIMIZATION: Use a semaphore to limit concurrent upsert tasks ---
        # A semaphore limits the number of concurrent tasks to avoid overwhelming the API.
        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_UPSERTS)

        async def upsert_batch(batch: List[Tuple]):
            async with semaphore:
                try:
                    # Use asyncio.to_thread to run the blocking upsert call
                    # This makes the main event loop non-blocking
                    await asyncio.to_thread(self.index.upsert, vectors=batch)
                    logger.debug(f"Successfully upserted a batch of {len(batch)} vectors.")
                except Exception as e:
                    logger.error(f"Failed to upsert batch: {e}")
        
        # Create a list of all the upsert tasks
        upsert_tasks = [upsert_batch(batch) for batch in vector_batches]
        
        # Run all tasks concurrently and wait for them to finish
        await asyncio.gather(*upsert_tasks)
        
        logger.info(f"Pinecone index build completed.")
    
    def search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Search for similar documents using query embedding."""
        if self.index is None:
            raise ValueError("Pinecone index not initialized.")
        
        if k is None:
            k = config.TOP_K_RESULTS
        
        query_hash = hashlib.md5(query.encode()).hexdigest()
        query_embedding = self._cached_create_embedding(query_hash, query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        search_results = []
        for match in results.matches:
            text_content = match.metadata.get('text', '')
            if not isinstance(text_content, str):
                logger.warning(f"Skipping non-string content from Pinecone match: {type(text_content)}")
                continue

            doc = Document(
                page_content=text_content,
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