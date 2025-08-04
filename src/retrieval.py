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
    def __init__(self):
        pass

    @lru_cache(maxsize=100)
    def _create_query_embedding(self, query: str, embeddings_manager: EmbeddingsManager) -> np.ndarray:
        embedding = embeddings_manager.create_embedding(query)
        return np.array([embedding], dtype=np.float32)

    def _boost_score_with_keywords(self, query: str, document_content: str, metadata: Dict[str, Any], base_score: float) -> float:
        query_words = set(query.lower().split())
        content_words = set(document_content.lower().split())
        overlap_count = len(query_words.intersection(content_words))
        keyword_boost = min(overlap_count * 0.15, 0.45)

        section_type = metadata.get('section_type', '').lower()
        metadata_boost = 0.0
        # Boost documents with specific, high-value content types
        if section_type in ['definition', 'eligibility', 'conditions', 'table']:
            metadata_boost = 0.2

        return base_score + keyword_boost + metadata_boost

    def _filter_and_rerank(self, search_results: List[Tuple[Document, float]], query: str, k: int) -> List[Dict[str, Any]]:
        final_results = []
        seen_content_hashes = set()

        for doc, base_score in search_results:
            content = doc.page_content
            metadata = doc.metadata
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
                'reasoning': f"Semantic similarity ({base_score:.3f}) + Keyword/Section boost"
            })

        final_results.sort(key=lambda x: x['confidence'], reverse=True)
        threshold = 0.5
        return [r for r in final_results if r['confidence'] > threshold][:k]

    def retrieve_relevant_chunks(self, query: str, embeddings_manager: EmbeddingsManager, k: int = 4) -> List[Dict[str, Any]]:
        if embeddings_manager.index is None:
            logger.error("No embeddings index available. Please load documents first.")
            return []

        try:
            logger.info(f"🔄 Starting enhanced retrieval for query: '{query}'")
            search_results = embeddings_manager.search(query=query, k=max(k * 3, 15))
            final_chunks = self._filter_and_rerank(search_results, query, k)

            if not final_chunks:
                logger.warning("⚠️ No relevant chunks found for the query.")
            else:
                logger.info(f"✅ Retrieved {len(final_chunks)} chunks.")

            return final_chunks

        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            return []

    async def retrieve_and_generate_answer_async(self, query: str, embeddings_manager: EmbeddingsManager) -> Tuple[str, List[str], float]:
        try:
            relevant_chunks = await asyncio.to_thread(self.retrieve_relevant_chunks, query, embeddings_manager, k=6)

            if not relevant_chunks:
                return "The policy document does not contain sufficient information to answer this question.", [], 0.0

            context_parts = [chunk['content'] for chunk in relevant_chunks[:6]]
            context = "\n\n".join(context_parts)

            # MODIFIED PROMPT
            prompt = f"""
You are an insurance expert assistant. Based on the policy excerpts provided, answer the user's question as accurately as possible. Focus on being concise and directly answering the user's query.

POLICY DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{query}

Guidelines:
- Start with a direct "Yes, it is covered" or "No, it is not covered" if a definitive answer is possible.
- If the policy mentions specific numbers (e.g., days, years, amounts), include them.
- Avoid unnecessary details and long descriptions.
- If a key detail is missing, state: "The policy does not provide sufficient information to answer this conclusively."
- Do NOT make up any information.

Final Answer:
"""

            client = OpenAI(api_key=config.OPENAI_API_KEY)
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=180,
                top_p=0.9
            )

            answer = response.choices[0].message.content.strip()
            confidence = max([c.get('confidence', 0.0) for c in relevant_chunks])
            sources = [c['metadata'].get('source', 'Unknown') for c in relevant_chunks]

            return answer, sources, confidence

        except Exception as e:
            logger.error(f"❌ Failed to generate answer: {e}")
            return "Error processing the question.", [], 0.0