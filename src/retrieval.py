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

    def _calculate_weighted_score(self, query: str, document_content: str, metadata: Dict[str, Any], base_similarity: float) -> Tuple[float, Dict[str, float]]:
        """Calculate weighted score combining similarity, keyword, and metadata components."""
        query_lower = query.lower()
        content_lower = document_content.lower()
        query_terms = query_lower.split()
        
        # Component 1: Keyword Relevance Score (0.0 - 1.0)
        keyword_score = self._calculate_keyword_score(query_lower, content_lower, query_terms)
        
        # Component 2: Contextual Metadata Score (0.0 - 1.0)
        metadata_score = self._calculate_contextual_metadata_score(query_lower, metadata)
        
        # Component 3: Proximity Score (0.0 - 1.0)
        proximity_score = self._calculate_proximity_score(query_terms, content_lower)
        
        # Weighted combination (prevents weak semantic matches from being rescued by coincidental keyword matches)
        weights = {
            'similarity': 0.5,    # Base semantic similarity
            'keyword': 0.25,      # Keyword relevance
            'metadata': 0.15,     # Section/context relevance
            'proximity': 0.1      # Term proximity
        }
        
        total_score = (
            base_similarity * weights['similarity'] +
            keyword_score * weights['keyword'] +
            metadata_score * weights['metadata'] +
            proximity_score * weights['proximity']
        )
        
        component_scores = {
            'similarity': base_similarity,
            'keyword': keyword_score,
            'metadata': metadata_score,
            'proximity': proximity_score
        }
        
        return min(total_score, 1.0), component_scores
    
    def _calculate_keyword_score(self, query_lower: str, content_lower: str, query_terms: List[str]) -> float:
        """Calculate keyword relevance score with exact phrase and term matching."""
        score = 0.0
        
        # Exact phrase match (highest value)
        if query_lower in content_lower:
            score += 0.6
        
        # Individual term matches
        term_matches = sum(1 for term in query_terms if term in content_lower)
        term_ratio = term_matches / len(query_terms) if query_terms else 0
        score += term_ratio * 0.3
        
        # Universal policy-relevant terms (applicable to any policy type)
        universal_policy_terms = {
            # Time-related terms
            'period': 0.1, 'duration': 0.1, 'time': 0.08, 'days': 0.08, 'months': 0.08, 'years': 0.08,
            # Conditions and requirements
            'condition': 0.1, 'requirement': 0.1, 'criteria': 0.1, 'eligibility': 0.1,
            # Limits and amounts
            'limit': 0.12, 'amount': 0.1, 'maximum': 0.1, 'minimum': 0.1, 'threshold': 0.1,
            # Coverage and benefits
            'coverage': 0.1, 'benefit': 0.1, 'entitlement': 0.1, 'provision': 0.1,
            # Exclusions and restrictions
            'exclusion': 0.1, 'exception': 0.1, 'restriction': 0.1, 'limitation': 0.1,
            # Process and procedures
            'procedure': 0.08, 'process': 0.08, 'application': 0.08, 'claim': 0.08,
            # Definitions and terms
            'definition': 0.08, 'means': 0.08, 'refers': 0.08, 'includes': 0.08
        }
        
        for term, weight in universal_policy_terms.items():
            if term in query_lower and term in content_lower:
                score += weight
        
        return min(score, 1.0)
    
    def _calculate_contextual_metadata_score(self, query_lower: str, metadata: Dict[str, Any]) -> float:
        """Calculate contextual relevance based on document section and query intent."""
        section_type = metadata.get('section_type', '').lower()
        chunk_strategy = metadata.get('chunk_strategy', '').lower()
        
        # Universal query intent classification (applicable to any policy type)
        numerical_keywords = ['limit', 'amount', 'percentage', 'percent', 'cost', 'fee', 'rate', 'value', 'number', 'quantity', 'maximum', 'minimum', 'threshold']
        definition_keywords = ['define', 'definition', 'what is', 'meaning of', 'refers to', 'means', 'includes', 'constitutes']
        exclusion_keywords = ['exclude', 'exclusion', 'not covered', 'exception', 'restriction', 'limitation', 'prohibited', 'forbidden']
        entitlement_keywords = ['entitle', 'entitled', 'benefit', 'provision', 'coverage', 'eligible', 'qualify', 'right to', 'access to']
        process_keywords = ['procedure', 'process', 'how to', 'steps', 'application', 'submit', 'request', 'claim']
        condition_keywords = ['condition', 'requirement', 'criteria', 'must', 'shall', 'need to', 'required to']
        
        query_intent = 'general'
        if any(kw in query_lower for kw in numerical_keywords):
            query_intent = 'numerical'
        elif any(kw in query_lower for kw in definition_keywords):
            query_intent = 'definition'
        elif any(kw in query_lower for kw in exclusion_keywords):
            query_intent = 'exclusion'
        elif any(kw in query_lower for kw in entitlement_keywords):
            query_intent = 'entitlement'
        elif any(kw in query_lower for kw in process_keywords):
            query_intent = 'process'
        elif any(kw in query_lower for kw in condition_keywords):
            query_intent = 'condition'
        
        # Universal section-intent matching scores
        intent_section_scores = {
            'numerical': {
                'table': 0.9, 'table_prose': 0.8, 'limits': 0.8, 'amounts': 0.9, 'rates': 0.9,
                'semantic_text': 0.3, 'definition': 0.2, 'general': 0.4
            },
            'definition': {
                'definition': 0.9, 'definitions': 0.9, 'semantic_text': 0.8, 'general': 0.7,
                'table': 0.2, 'table_prose': 0.3, 'glossary': 0.9
            },
            'exclusion': {
                'exclusion': 0.9, 'exclusions': 0.9, 'restrictions': 0.8, 'limitations': 0.8,
                'conditions': 0.7, 'semantic_text': 0.7, 'table': 0.3, 'definition': 0.4
            },
            'entitlement': {
                'benefits': 0.9, 'entitlements': 0.9, 'provisions': 0.8, 'coverage': 0.8,
                'eligibility': 0.8, 'semantic_text': 0.7, 'exclusion': 0.2, 'definition': 0.5
            },
            'process': {
                'procedure': 0.9, 'procedures': 0.9, 'process': 0.9, 'application': 0.8,
                'steps': 0.8, 'semantic_text': 0.7, 'table': 0.4, 'definition': 0.5
            },
            'condition': {
                'conditions': 0.9, 'requirements': 0.9, 'criteria': 0.8, 'eligibility': 0.8,
                'semantic_text': 0.7, 'table': 0.5, 'definition': 0.6
            },
            'general': {
                'semantic_text': 0.7, 'general': 0.8, 'definition': 0.6,
                'table': 0.4, 'table_prose': 0.5, 'overview': 0.7
            }
        }
        
        # Get score based on section type and chunk strategy
        section_scores = intent_section_scores.get(query_intent, intent_section_scores['general'])
        
        score = 0.0
        if section_type in section_scores:
            score = max(score, section_scores[section_type])
        if chunk_strategy in section_scores:
            score = max(score, section_scores[chunk_strategy])
        
        return score
    
    def _calculate_proximity_score(self, query_terms: List[str], content_lower: str) -> float:
        """Calculate proximity score based on how close query terms appear to each other."""
        if len(query_terms) < 2:
            return 0.5  # Neutral score for single terms
        
        score = 0.0
        content_words = content_lower.split()
        
        # Find positions of query terms in content
        term_positions = {}
        for term in query_terms:
            positions = [i for i, word in enumerate(content_words) if term in word]
            if positions:
                term_positions[term] = positions
        
        if len(term_positions) < 2:
            return 0.3  # Low score if terms don't co-occur
        
        # Calculate minimum distance between any two query terms
        min_distance = float('inf')
        terms_with_positions = list(term_positions.items())
        
        for i, (term1, pos1_list) in enumerate(terms_with_positions):
            for j, (term2, pos2_list) in enumerate(terms_with_positions[i+1:], i+1):
                for pos1 in pos1_list:
                    for pos2 in pos2_list:
                        distance = abs(pos1 - pos2)
                        min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            return 0.3
        
        # Convert distance to score (closer = higher score)
        if min_distance <= 3:
            score = 1.0  # Very close
        elif min_distance <= 10:
            score = 0.8  # Close
        elif min_distance <= 25:
            score = 0.6  # Moderate
        elif min_distance <= 50:
            score = 0.4  # Distant
        else:
            score = 0.2  # Very distant
        
        return score

    def _filter_and_rerank(self, search_results: List[Tuple[Document, float]], query: str, k: int) -> List[Dict[str, Any]]:
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

            # Calculate weighted score
            weighted_score, component_scores = self._calculate_weighted_score(
                query, content, metadata, base_similarity
            )
            
            # COST OPTIMIZATION: Add content quality scoring
            content_lower = content.lower()
            quality_bonus = 0.0
            
            # Bonus for chunks with multiple query terms
            term_matches = sum(1 for term in query_terms if term in content_lower)
            if term_matches >= 2:
                quality_bonus += 0.1
            
            # Bonus for chunks with numbers (likely contain specific details)
            if any(char.isdigit() for char in content):
                quality_bonus += 0.05
            
            # Bonus for chunks with policy-specific indicators
            policy_indicators = ['period', 'coverage', 'limit', 'condition', 'benefit', 'exclusion']
            if any(indicator in content_lower for indicator in policy_indicators):
                quality_bonus += 0.05
            
            final_score = min(weighted_score + quality_bonus, 1.0)

            # Create detailed reasoning for transparency
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

        # Sort by final confidence score
        final_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # COST OPTIMIZATION: Smart filtering for quality chunks
        if final_results:
            max_score = final_results[0]['confidence']
            
            # More aggressive filtering for high-quality chunks
            high_quality_threshold = max(0.4, max_score * 0.7)
            high_quality_chunks = [
                result for result in final_results 
                if result['confidence'] >= high_quality_threshold
            ]
            
            # If we have enough high-quality chunks, use them
            if len(high_quality_chunks) >= 6:
                filtered_results = high_quality_chunks[:8]  # Top 8 high-quality chunks
            else:
                # Otherwise, use relaxed threshold
                relaxed_threshold = max(0.3, max_score * 0.5)
                filtered_results = [
                    result for result in final_results 
                    if result['confidence'] >= relaxed_threshold
                ]
                
                # Ensure minimum coverage
                if len(filtered_results) < 6:
                    filtered_results = final_results[:8]
            
            # Cap at reasonable number for cost control
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
            # Cast wider net for initial search to ensure we don't miss relevant content
            search_results = embeddings_manager.search(query=query, k=max(k * 5, 25))
            final_chunks = self._filter_and_rerank(search_results, query, k)

            if not final_chunks:
                logger.warning("⚠️ No relevant chunks found for the query.")
                # Try a more relaxed search if no results found
                logger.info("🔄 Attempting relaxed search...")
                search_results = embeddings_manager.search(query=query, k=50)
                final_chunks = self._filter_and_rerank_relaxed(search_results, query, k)
            else:
                logger.info(f"✅ Retrieved {len(final_chunks)} chunks.")

            return final_chunks

        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            return []

    def _expand_query_for_better_retrieval(self, query: str) -> List[str]:
        """Cost-optimized universal query expansion for better retrieval."""
        query_lower = query.lower()
        expanded_queries = [query]  # Always include original query
        
        # COST OPTIMIZATION: Targeted expansion based on query intent
        query_words = query_lower.split()
        
        # Universal semantic expansion patterns (domain-agnostic)
        semantic_patterns = {
            # Time-related terms
            'period': ['duration', 'time', 'timeframe'],
            'waiting': ['delay', 'period', 'time'],
            'grace': ['extension', 'additional', 'extra'],
            
            # Coverage/benefit terms
            'coverage': ['benefit', 'provision', 'entitlement'],
            'benefit': ['coverage', 'provision', 'allowance'],
            'limit': ['maximum', 'cap', 'threshold'],
            'discount': ['reduction', 'deduction', 'savings'],
            
            # Process/procedure terms
            'procedure': ['process', 'treatment', 'operation'],
            'surgery': ['operation', 'procedure', 'treatment'],
            'treatment': ['therapy', 'procedure', 'care'],
            
            # Institutional terms
            'hospital': ['facility', 'institution', 'center'],
            'facility': ['institution', 'center', 'hospital'],
            
            # Condition/requirement terms
            'condition': ['requirement', 'criteria', 'prerequisite'],
            'requirement': ['condition', 'criteria', 'prerequisite'],
            'exclusion': ['exception', 'restriction', 'limitation']
        }
        
        # Universal policy terms for broader queries
        universal_terms = {
            'period': ['time', 'duration'],
            'limit': ['maximum', 'cap'],
            'coverage': ['benefit', 'provision'],
            'exclusion': ['exception', 'restriction'],
            'definition': ['meaning', 'refers to']
        }
        
        # COST OPTIMIZATION: Universal semantic expansion
        expansions = []
        
        # Priority 1: Semantic pattern expansions (universal approach)
        for word in query_words:
            if word in semantic_patterns:
                # Replace with best semantic synonym
                best_synonym = semantic_patterns[word][0]  # Use most relevant synonym
                expansion = query.replace(word, best_synonym)
                if expansion not in expanded_queries and expansion != query:
                    expansions.append(expansion)
                break  # Only expand one term to control cost
        
        # Priority 2: Universal term synonyms (if no critical terms found)
        if not expansions:
            for word in query_words:
                if word in universal_terms:
                    for synonym in universal_terms[word][:1]:  # Only use best synonym
                        expansion = query.replace(word, synonym)
                        if expansion not in expanded_queries:
                            expansions.append(expansion)
                            break
                    break
        
        # Priority 3: Number format variations (for specific queries)
        import re
        if re.search(r'\b\d+\b', query) and not expansions:
            number_patterns = re.findall(r'\b\d+\b', query)
            for num in number_patterns[:1]:  # Only expand first number
                variations = [
                    query.replace(num, f"{num} days"),
                    query.replace(num, f"{num} months")
                ]
                for variation in variations[:1]:  # Only use best variation
                    if variation != query and variation not in expanded_queries:
                        expansions.append(variation)
                        break
                break
        
        # COST CONTROL: Limit to maximum 2 total queries (original + 1 expansion)
        if expansions:
            expanded_queries.append(expansions[0])  # Only add the best expansion
        
        return expanded_queries[:2]  # Maximum 2 queries total for cost efficiency
    
    async def retrieve_and_generate_answer_async(self, query: str, embeddings_manager: EmbeddingsManager) -> Tuple[str, List[str], float]:
        try:
            # Expand query for better retrieval
            expanded_queries = self._expand_query_for_better_retrieval(query)
            
            # Retrieve chunks for all expanded queries
            all_chunks = []
            for expanded_query in expanded_queries:
                chunks = await asyncio.to_thread(self.retrieve_relevant_chunks, expanded_query, embeddings_manager, k=8)
                all_chunks.extend(chunks)
            
            # Remove duplicates and get top chunks
            seen_hashes = set()
            unique_chunks = []
            for chunk in all_chunks:
                chunk_hash = hash(chunk['content'][:150])
                if chunk_hash not in seen_hashes:
                    seen_hashes.add(chunk_hash)
                    unique_chunks.append(chunk)
            
            # Sort by confidence and take top chunks
            unique_chunks.sort(key=lambda x: x['confidence'], reverse=True)
            relevant_chunks = unique_chunks[:10]

            if not relevant_chunks:
                return "The policy does not provide sufficient information to answer this conclusively.", [], 0.0

            # Use more context for better accuracy
            context_parts = [chunk['content'] for chunk in relevant_chunks[:8]]
            context = "\n\n".join(context_parts)

            # UNIVERSAL FACTUAL ANSWER PROMPT
            prompt = f"""
Analyze the policy excerpts and provide a direct, factual answer to the question.

POLICY EXCERPTS:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. PROVIDE DIRECT FACTS: Give a clear, factual statement that directly answers the question.

2. INCLUDE ALL KEY DETAILS: Pack all essential information into one comprehensive sentence:
   • Exact numbers, percentages, time periods
   • All conditions, requirements, and limitations
   • Specific acts, regulations, or legal references
   • Important exceptions or qualifications

3. USE MOST ACCURATE INFORMATION: If you find conflicting details, use the most specific and authoritative version.

4. PROFESSIONAL TONE: Use clear, professional language without unnecessary prefixes like "Yes" or "No" unless the question specifically asks for confirmation.

5. COMPLETE COVERAGE: Ensure your answer addresses all aspects of the question comprehensively.

FORMAT EXAMPLES:
- "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
- "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
- "The policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy, requiring the female insured person to have been continuously covered for at least 24 months with the benefit limited to two deliveries or terminations during the policy period."

Answer:
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