import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from functools import lru_cache
from openai import OpenAI
import asyncio

from .embeddings import EmbeddingsManager
from config import config

logger = logging.getLogger(__name__)

class RetrievalSystem:
    """
    A robust and universal retrieval system for RAG, designed to work with any document.
    This system combines semantic search with a dynamic keyword-based booster for superior accuracy.
    """

    def __init__(self):
        """Initializes the RetrievalSystem without a specific embeddings_manager.
        The embeddings manager will be passed to the retrieval method directly."""
        pass

    @lru_cache(maxsize=100)
    def _create_query_embedding(self, query: str, embeddings_manager: EmbeddingsManager) -> np.ndarray:
        """Create and cache the embedding for a given query."""
        embedding = embeddings_manager.create_embedding(query)
        return np.array([embedding], dtype=np.float32)

    def _get_document_content(self, idx: int, embeddings_manager: EmbeddingsManager) -> str:
        """Get content from the documents list, with a safety check."""
        if idx < len(embeddings_manager.documents):
            return embeddings_manager.documents[idx].page_content
        return ""

    def _get_document_metadata(self, idx: int, embeddings_manager: EmbeddingsManager) -> Dict[str, Any]:
        """Get metadata from the documents list, with a safety check."""
        if idx < len(embeddings_manager.documents):
            return embeddings_manager.documents[idx].metadata
        return {}

    def _boost_score_with_keywords(self, query: str, document_content: str, metadata: Dict[str, Any], base_score: float) -> float:
        """Dynamically boost the retrieval score based on keyword overlap and metadata."""
        query_words = set(query.lower().split())
        content_words = set(document_content.lower().split())
        
        overlap_count = len(query_words.intersection(content_words))
        keyword_boost = min(overlap_count * 0.15, 0.45)
        
        length_penalty = 0
        if len(document_content) > 1000:
            length_penalty = -0.1
        elif len(document_content) > 1500:
            length_penalty = -0.2
        
        metadata_boost = 0
        content_type = metadata.get('content_type', 'general')
        if content_type == 'definition':
            metadata_boost = 0.2
        elif content_type == 'policy':
            metadata_boost = 0.1
        
        return base_score + keyword_boost + length_penalty + metadata_boost

    def _filter_and_rerank(self, search_results: List[tuple], query: str, k: int, embeddings_manager: EmbeddingsManager) -> List[Dict[str, Any]]:
        """Filter, enhance, and rerank search results."""
        final_results = []
        seen_content_hashes = set()
        
        for idx, base_score in search_results:
            content = self._get_document_content(idx, embeddings_manager)
            metadata = self._get_document_metadata(idx, embeddings_manager)
            if not content:
                continue

            content_hash = hash(content.strip()[:150])
            if content_hash in seen_content_hashes:
                continue
            seen_content_hashes.add(content_hash)

            enhanced_score = self._boost_score_with_keywords(query, content, metadata, base_score)
            
            final_results.append({
                'content': content,
                'metadata': metadata,
                'similarity': float(base_score),
                'confidence': min(enhanced_score, 1.0),
                'reasoning': f"Semantic similarity ({base_score:.3f}) + Keyword boost"
            })
            
        final_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        threshold = 0.5
        filtered_results = [r for r in final_results if r['confidence'] > threshold]

        return filtered_results[:k]

    def retrieve_relevant_chunks(self, query: str, embeddings_manager: EmbeddingsManager, k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant chunks using a semantic and keyword-boosted approach.
        This method is optimized for accuracy and relevance.
        """
        if embeddings_manager.index is None:
            logger.error("No embeddings index available. Please load documents first.")
            return []

        try:
            logger.info(f"🔄 Starting universal retrieval for query: '{query}'")
            
            search_k = max(k * 3, 10)
            query_vector = self._create_query_embedding(query, embeddings_manager)
            scores, indices = embeddings_manager.index.search(query_vector, search_k)
            
            search_results = list(zip(indices[0], scores[0]))
            
            final_chunks = self._filter_and_rerank(search_results, query, k, embeddings_manager)
            
            if not final_chunks:
                logger.warning(f"⚠️ No relevant chunks found for the query.")
            else:
                logger.info(f"✅ Successfully retrieved {len(final_chunks)} optimized chunks.")

            return final_chunks
            
        except Exception as e:
            logger.error(f"❌ Error in retrieval process: {e}")
            return []
            
    async def retrieve_and_generate_answer_async(self, query: str, embeddings_manager: EmbeddingsManager) -> Tuple[str, List[str], float]:
        """
        Performs retrieval and then generates an answer asynchronously.
        """
        try:
            # Perform retrieval
            relevant_chunks = await asyncio.to_thread(self.retrieve_relevant_chunks, query, embeddings_manager, k=4)
            
            if not relevant_chunks:
                return "The policy document does not contain information on this topic.", [], 0.0
            
            # Prepare context for LLM
            context_parts = [chunk['content'] for chunk in relevant_chunks[:4]]
            context = "\n\n".join(context_parts)
            
            # New, better prompt for robust answer generation
            prompt = f"""You are an insurance policy expert. Your task is to provide a comprehensive and accurate answer to the user's question based ONLY on the provided policy text.

POLICY DOCUMENT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Do not make up any information.
- If the policy text provides a specific number (e.g., '30 days', '24 months', '5%'), include that number in your answer.
- If the policy mentions conditions, exclusions, or limitations, summarize them clearly.
- If the policy text does not contain the answer, state that "The policy document does not contain information on this topic."

Answer:"""
            
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=150,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            confidence = max([c.get('confidence', 0.0) for c in relevant_chunks]) if relevant_chunks else 0.0
            sources = [c['metadata'].get('source', 'Unknown') for c in relevant_chunks]
            
            return answer, sources, confidence
        
        except Exception as e:
            logger.error(f"Failed to generate answer for query '{query}': {e}")
            return "Error processing this question.", [], 0.0