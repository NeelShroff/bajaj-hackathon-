import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from functools import lru_cache
import asyncio
from openai import OpenAI, AsyncOpenAI
import tiktoken

from .embeddings import EmbeddingsManager
from config import config

logger = logging.getLogger(__name__)

# Use async version of LLM client
async_llm_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback for models not in tiktoken registry
        return len(text) // 4

class RetrievalSystem:
    def __init__(self):
        pass

    @lru_cache(maxsize=100)
    def _create_query_embedding(self, query: str, embeddings_manager: EmbeddingsManager) -> np.ndarray:
        embedding = embeddings_manager.create_embedding(query)
        return np.array([embedding], dtype=np.float32)

    # ... (All other helper methods like _calculate_weighted_score, _filter_and_rerank remain the same) ...
    def _calculate_weighted_score(self, query: str, document_content: str, metadata: Dict[str, Any], base_similarity: float) -> Tuple[float, Dict[str, float]]:
        """Calculate weighted score combining similarity, keyword, and metadata components."""
        query_lower = query.lower()
        content_lower = document_content.lower()
        query_terms = query_lower.split()
        
        keyword_score = self._calculate_keyword_score(query_lower, content_lower, query_terms)
        metadata_score = self._calculate_contextual_metadata_score(query_lower, metadata)
        proximity_score = self._calculate_proximity_score(query_terms, content_lower)
        
        weights = {'similarity': 0.5, 'keyword': 0.25, 'metadata': 0.15, 'proximity': 0.1}
        
        total_score = (
            base_similarity * weights['similarity'] +
            keyword_score * weights['keyword'] +
            metadata_score * weights['metadata'] +
            proximity_score * weights['proximity']
        )
        
        component_scores = {'similarity': base_similarity, 'keyword': keyword_score, 'metadata': metadata_score, 'proximity': proximity_score}
        
        return min(total_score, 1.0), component_scores
    
    def _calculate_keyword_score(self, query_lower: str, content_lower: str, query_terms: List[str]) -> float:
        """Calculate keyword relevance score with exact phrase and term matching."""
        score = 0.0
        if query_lower in content_lower:
            score += 0.6
        term_matches = sum(1 for term in query_terms if term in content_lower)
        term_ratio = term_matches / len(query_terms) if query_terms else 0
        score += term_ratio * 0.3
        
        universal_terms = {'period', 'duration', 'time', 'days', 'months', 'years', 'condition', 'requirement', 
                           'criteria', 'eligibility', 'limit', 'amount', 'maximum', 'minimum', 'threshold',
                           'coverage', 'benefit', 'entitlement', 'provision', 'exclusion', 'exception', 
                           'restriction', 'limitation', 'procedure', 'process', 'application', 'claim', 
                           'definition', 'means', 'refers'}
        
        for term in universal_terms:
            if term in query_lower and term in content_lower:
                score += 0.05
        
        return min(score, 1.0)
    
    def _calculate_contextual_metadata_score(self, query_lower: str, metadata: Dict[str, Any]) -> float:
        """Calculate contextual relevance based on document section and query intent."""
        section_type = metadata.get('section_type', '').lower()
        chunk_strategy = metadata.get('chunk_strategy', '').lower()
        
        query_intent_keywords = {
            'numerical': ['limit', 'amount', 'cost', 'rate', 'maximum'],
            'definition': ['define', 'meaning of', 'what is'],
            'exclusion': ['exclude', 'not covered', 'exception'],
            'entitlement': ['benefit', 'coverage', 'eligible'],
            'process': ['procedure', 'how to', 'steps'],
            'condition': ['condition', 'requirement', 'criteria'],
        }
        
        query_intent = 'general'
        for intent, keywords in query_intent_keywords.items():
            if any(kw in query_lower for kw in keywords):
                query_intent = intent
                break
        
        score_mapping = {
            'numerical': {'table': 0.9, 'section_type_rate': 0.8},
            'definition': {'definition': 0.9, 'glossary': 0.9},
            'exclusion': {'exclusion': 0.9, 'restrictions': 0.8},
            'entitlement': {'benefits': 0.9, 'provisions': 0.8},
            'process': {'procedure': 0.9, 'process': 0.9},
            'condition': {'conditions': 0.9, 'requirements': 0.9},
            'general': {'general': 0.7, 'semantic_text': 0.8}
        }
        
        section_scores = score_mapping.get(query_intent, score_mapping['general'])
        
        score = 0.0
        if section_type in section_scores:
            score = max(score, section_scores[section_type])
        if chunk_strategy in section_scores:
            score = max(score, section_scores[chunk_strategy])
        
        return score

    def _calculate_proximity_score(self, query_terms: List[str], content_lower: str) -> float:
        """Calculate proximity score based on how close query terms appear to each other."""
        if len(query_terms) < 2:
            return 0.5
        
        score = 0.0
        content_words = content_lower.split()
        
        term_positions = {}
        for term in query_terms:
            positions = [i for i, word in enumerate(content_words) if term in word]
            if positions:
                term_positions[term] = positions
        
        if len(term_positions) < 2:
            return 0.3
        
        min_distance = float('inf')
        terms_with_positions = list(term_positions.items())
        
        for i, (_, pos1_list) in enumerate(terms_with_positions):
            for j, (_, pos2_list) in enumerate(terms_with_positions[i+1:], i+1):
                for pos1 in pos1_list:
                    for pos2 in pos2_list:
                        distance = abs(pos1 - pos2)
                        min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            return 0.3
        
        if min_distance <= 3:
            score = 1.0
        elif min_distance <= 10:
            score = 0.8
        elif min_distance <= 25:
            score = 0.6
        elif min_distance <= 50:
            score = 0.4
        else:
            score = 0.2
        
        return score

    def _filter_and_rerank(self, search_results: List[Tuple[Document, float]], query: str) -> List[Dict[str, Any]]:
        """Cost-optimized filter and rerank with smart chunk selection."""
        final_results = []
        seen_content_hashes = set()
        query_terms = set(query.lower().split())

        for doc, base_similarity in search_results:
            content = doc.page_content
            metadata = doc.metadata
            if not content:
                continue

            content_hash = hash(content.strip()[:150])
            if content_hash in seen_content_hashes:
                continue
            seen_content_hashes.add(content_hash)

            weighted_score, component_scores = self._calculate_weighted_score(
                query, content, metadata, base_similarity
            )
            
            content_lower = content.lower()
            quality_bonus = 0.0
            
            term_matches = sum(1 for term in query_terms if term in content_lower)
            if term_matches >= 2:
                quality_bonus += 0.1
            
            if any(char.isdigit() for char in content):
                quality_bonus += 0.05
            
            document_indicators = ['period', 'coverage', 'limit', 'condition', 'benefit', 'exclusion', 
                                   'clause', 'section', 'part', 'article', 'policy']
            if any(indicator in content_lower for indicator in document_indicators):
                quality_bonus += 0.05
            
            final_score = min(weighted_score + quality_bonus, 1.0)

            reasoning_parts = []
            for component, score in component_scores.items():
                reasoning_parts.append(f"{component}: {score:.3f}")
            if quality_bonus > 0:
                reasoning_parts.append(f"quality: +{quality_bonus:.3f}")
            reasoning = f"Final score: {final_score:.3f} ({', '.join(reasoning_parts)})"

            final_results.append({
                'content': content,
                'metadata': metadata,
                'similarity': float(base_similarity),
                'confidence': final_score,
                'component_scores': component_scores,
                'quality_bonus': quality_bonus,
                'reasoning': reasoning
            })

        final_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        if final_results:
            max_score = final_results[0]['confidence']
            
            high_quality_threshold = max(0.4, max_score * 0.7)
            high_quality_chunks = [result for result in final_results if result['confidence'] >= high_quality_threshold]
            
            if len(high_quality_chunks) >= 6:
                filtered_results = high_quality_chunks[:8]
            else:
                relaxed_threshold = max(0.3, max_score * 0.5)
                filtered_results = [result for result in final_results if result['confidence'] >= relaxed_threshold]
                
                if len(filtered_results) < 6:
                    filtered_results = final_results[:8]
            
            filtered_results = filtered_results[:10]
        else:
            filtered_results = final_results
        
        logger.info(f"📊 Cost-Optimized Reranking: {len(search_results)} → {len(filtered_results)} chunks")
        return filtered_results

    def retrieve_relevant_chunks(self, query: str, embeddings_manager: EmbeddingsManager, k: int = 4) -> List[Dict[str, Any]]:
        if embeddings_manager.index is None:
            logger.error("No embeddings index available. Please load documents first.")
            return []

        try:
            logger.info(f"🔄 Starting enhanced retrieval for query: '{query}'")
            search_results = embeddings_manager.search(query=query, k=max(k * 5, 25))
            final_chunks = self._filter_and_rerank(search_results, query)
            
            if not final_chunks:
                logger.warning("⚠️ No relevant chunks found for the query.")
            else:
                logger.info(f"✅ Retrieved {len(final_chunks)} chunks.")

            return final_chunks

        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            return []

    async def retrieve_and_generate_answer_async(self, queries: List[str], embeddings_manager: EmbeddingsManager, original_query: str) -> Tuple[str, List[str], float]:
        MAX_CONTEXT_TOKENS = 12000
        
        try:
            token_usage = {'embedding_tokens': 0, 'llm_input_tokens': 0, 'llm_output_tokens': 0, 'total_tokens': 0}
            
            # Retrieve all chunks for all expanded queries concurrently
            query_tasks = [asyncio.to_thread(self.retrieve_relevant_chunks, query, embeddings_manager, k=8) for query in queries]
            query_results = await asyncio.gather(*query_tasks)
            
            # Flatten all chunks from all queries
            all_chunks = [chunk for chunks in query_results for chunk in chunks]
            
            seen_hashes = set()
            unique_chunks = []
            for chunk in all_chunks:
                chunk_hash = hash(chunk['content'][:150])
                if chunk_hash not in seen_hashes:
                    seen_hashes.add(chunk_hash)
                    unique_chunks.append(chunk)
            
            unique_chunks.sort(key=lambda x: x['confidence'], reverse=True)
            relevant_chunks = unique_chunks[:10]

            if not relevant_chunks:
                return "The document does not provide sufficient information to answer this question.", [], 0.0

            context_parts = []
            current_context_tokens = 0
            for chunk in relevant_chunks:
                chunk_tokens = count_tokens(chunk['content'], config.OPENAI_MODEL)
                if current_context_tokens + chunk_tokens <= MAX_CONTEXT_TOKENS:
                    context_parts.append(chunk['content'])
                    current_context_tokens += chunk_tokens
                else:
                    logger.warning(f"❌ Dropping chunk to avoid token limit. Current context tokens: {current_context_tokens}, Chunk tokens: {chunk_tokens}")
                    break
            
            if not context_parts:
                return "The relevant information is too long to process.", [], 0.0

            context = "\n\n".join(context_parts)

            prompt = f"""
Analyze the provided document excerpts and give a direct, factual answer to the user's question.

DOCUMENT EXCERPTS:
{context}

QUESTION: {original_query}

INSTRUCTIONS:
1. FORMULATE THE ANSWER IN A SINGLE, COMPREHENSIVE SENTENCE OR A SHORT PARAGRAPH.
2. INCLUDE ALL SPECIFIC DETAILS from the document, such as exact numbers, names, dates, amounts, time periods, and legal references.
3. CITE THE SOURCE by mentioning the clause, section, or policy name if relevant.
4. If the answer is "yes" or "no", start with that word but immediately follow with the justification from the text.
5. If the information is NOT present in the provided excerpts, state this clearly and concisely, like "The document does not specify...".

Answer:
"""

            token_usage['llm_input_tokens'] = count_tokens(prompt, config.OPENAI_MODEL)
            
            # --- OPTIMIZATION: Use the asynchronous LLM client directly ---
            response = await async_llm_client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=180,
                top_p=0.9
            )

            answer = response.choices[0].message.content.strip()
            
            token_usage['llm_output_tokens'] = count_tokens(answer, config.OPENAI_MODEL)
            
            token_usage['total_tokens'] = token_usage['embedding_tokens'] + token_usage['llm_input_tokens'] + token_usage['llm_output_tokens']
            
            logger.info(f"🔢 TOKEN USAGE - Query: '{original_query[:50]}...'")
            logger.info(f"   📊 Embedding tokens: {token_usage['embedding_tokens']}")
            logger.info(f"   📥 LLM input tokens: {token_usage['llm_input_tokens']}")
            logger.info(f"   📤 LLM output tokens: {token_usage['llm_output_tokens']}")
            logger.info(f"   💰 TOTAL tokens: {token_usage['total_tokens']}")
            
            confidence = max([c.get('confidence', 0.0) for c in relevant_chunks] or [0.0])
            sources = [c['metadata'].get('source', 'Unknown') for c in relevant_chunks]

            return answer, sources, confidence

        except Exception as e:
            logger.error(f"❌ Failed to generate answer: {e}")
            return "Error processing the question.", [], 0.0