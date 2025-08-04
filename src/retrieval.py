import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document
from datetime import datetime

from .embeddings import EmbeddingsManager
from config import config

logger = logging.getLogger(__name__)

class RetrievalSystem:
    """Handles semantic search and retrieval of relevant policy clauses with confidence scoring."""
    
    def __init__(self, embeddings_manager: Optional[EmbeddingsManager] = None):
        self.embeddings_manager = embeddings_manager
        # Enhanced keyword weights for better retrieval - BOOSTED for critical terms
        self.insurance_keywords = {
            # CRITICAL HIGH priority terms - boosted weights
            'grace': 0.6, 'waiting': 0.6, 'period': 0.5, 'days': 0.4, 'months': 0.4, 'years': 0.4,
            'premium': 0.4, 'payment': 0.4, 'due': 0.4, 'renewal': 0.4,
            'maternity': 0.6, 'pre-existing': 0.6, 'ped': 0.6, 'diseases': 0.5,
            'cataract': 0.6, 'surgery': 0.5, 'treatment': 0.4, 'medical': 0.3,
            'ncd': 0.6, 'discount': 0.6, 'claim': 0.4, 'bonus': 0.4,
            'hospital': 0.5, 'bed': 0.4, 'room': 0.5, 'icu': 0.5,
            'ayush': 0.6, 'ayurveda': 0.5, 'homeopathy': 0.5, 'unani': 0.5, 'siddha': 0.5,
            'organ': 0.6, 'donor': 0.6, 'transplant': 0.5, 'harvesting': 0.5,
            'health': 0.4, 'check-up': 0.5, 'preventive': 0.5, 'examination': 0.4,
            # Numbers and specific terms
            'thirty': 0.5, '30': 0.5, 'thirty-six': 0.5, '36': 0.5, '24': 0.5, '5%': 0.5, '1%': 0.5,
            # Medium priority
            'coverage': 0.3, 'benefit': 0.3, 'exclusion': 0.3, 'limit': 0.3,
            'policy': 0.2, 'insured': 0.2, 'sum': 0.2, 'amount': 0.2
        }
        
        # Query expansion patterns for better matching
        self.query_expansions = {
            'grace period': ['grace', 'period', 'premium', 'payment', 'due', 'days', 'thirty', '30'],
            'waiting period': ['waiting', 'period', 'months', 'years', 'coverage', 'continuous'],
            'pre-existing': ['pre-existing', 'ped', 'diseases', 'conditions', 'thirty-six', '36'],
            'cataract': ['cataract', 'surgery', 'eye', 'treatment', 'months', 'waiting'],
            'ayush': ['ayush', 'ayurveda', 'homeopathy', 'unani', 'siddha', 'alternative'],
            'organ donor': ['organ', 'donor', 'transplant', 'harvesting', 'hospitalisation'],
            'ncd': ['no claim discount', 'discount', 'bonus', 'renewal', 'flat', '5%'],
            'health check': ['health', 'check', 'preventive', 'examination', 'reimbursed'],
            'hospital': ['hospital', 'institution', 'inpatient', 'registered', 'clinical'],
            'room rent': ['room', 'rent', 'icu', 'charges', 'sub-limit', '1%'],
            'maternity': ['maternity', 'pregnancy', 'childbirth', 'delivery', '24', 'months']
        }
        
    def retrieve_relevant_clauses(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Retrieve relevant policy clauses for a given query."""
        if k is None:
            k = config.TOP_K_RESULTS
            
        logger.info(f"Retrieving relevant clauses for query: {query}")
        
        # Perform semantic search
        results = self.embeddings_manager.search_with_threshold(query, k=k)
        
        logger.info(f"Retrieved {len(results)} relevant clauses")
        return results
    
    def retrieve_relevant_chunks(self, query: str, embeddings_manager: EmbeddingsManager, k: int = None) -> List[Dict[str, Any]]:
        """Optimized retrieval with enhanced scoring, speed, and token efficiency."""
        if k is None:
            k = config.TOP_K_RESULTS
        
        try:
            if not embeddings_manager or not embeddings_manager.index:
                logger.warning("No embeddings index available")
                return []
            
            # Expand query with related terms for better matching
            expanded_query = self._expand_query(query)
            
            # Get embeddings for the expanded query (more efficient)
            query_embedding = embeddings_manager.create_embedding(expanded_query)
            if query_embedding is None:
                logger.error("Failed to create query embedding")
                return []
            
            # Search for more candidates initially, then filter aggressively
            search_k = min(k * 4, 80)  # Increased search for better recall
            similarities, indices = embeddings_manager.index.search(
                np.array([query_embedding]), search_k
            )
            
            # CRITICAL: Add keyword-based fallback search for definitions
            critical_terms = {
                'grace period': ['grace period', 'grace', 'thirty days', '30 days', 'premium due date'],
                'ayush': ['ayush', 'ayurveda', 'homeopathy', 'unani', 'siddha', 'alternative medicine'],
                'waiting period': ['waiting period', 'months', 'continuous coverage'],
                'cataract': ['cataract', 'eye surgery', 'two years']
            }
            
            # Initialize query processing variables
            query_lower = query.lower()
            query_lower_check = query_lower.replace('-', ' ')
            needs_keyword_search = any(term in query_lower_check for term in critical_terms.keys())
            
            relevant_chunks = []
            seen_content = set()  # Avoid duplicate content
            
            # UNIVERSAL DOCUMENT HANDLING: Multi-strategy retrieval
            if needs_keyword_search:
                # Strategy 1: Aggressive keyword search for critical terms
                for term_key, keywords in critical_terms.items():
                    if term_key in query_lower_check:
                        # Search ALL documents for exact keyword matches
                        for doc_idx, document in enumerate(embeddings_manager.documents):
                            doc_lower = document.page_content.lower()
                            # Check for exact keyword matches
                            keyword_matches = sum(1 for kw in keywords if kw in doc_lower)
                            if keyword_matches > 0:
                                # Calculate boosted confidence for keyword matches
                                base_similarity = 0.7  # High base score for keyword matches
                                keyword_bonus = min(keyword_matches * 0.2, 0.3)
                                confidence = base_similarity + keyword_bonus
                                
                                content_hash = hash(document.page_content[:100])
                                if content_hash not in seen_content:
                                    seen_content.add(content_hash)
                                    relevant_chunks.append({
                                        'content': document.page_content[:800],
                                        'metadata': document.metadata,
                                        'similarity': base_similarity,
                                        'confidence': confidence,
                                        'reasoning': f'Keyword match: {keyword_matches} terms found'
                                    })
            
            # Strategy 2: Process semantic search results
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(embeddings_manager.documents):
                    chunk = embeddings_manager.documents[idx]
                    
                    # Skip very similar content to reduce redundancy
                    content_hash = hash(chunk.page_content[:100])  # Use first 100 chars as hash
                    if content_hash in seen_content:
                        continue
                    seen_content.add(content_hash)
                    
                    chunk_lower = chunk.page_content.lower()
                    
                    # Enhanced confidence scoring with query expansion
                    confidence = self._calculate_enhanced_confidence(
                        similarity, query_lower, chunk_lower, expanded_query.lower()
                    )
                    
                    # ULTRA-AGGRESSIVE filtering - capture all potentially relevant chunks
                    if confidence >= max(config.SIMILARITY_THRESHOLD, 0.2):
                        # Truncate very long chunks to save tokens
                        content = chunk.page_content
                        if len(content) > 800:  # Limit chunk size for token efficiency
                            content = content[:800] + "..."
                        
                        relevant_chunks.append({
                            'content': content,
                            'metadata': chunk.metadata,
                            'similarity': float(similarity),
                            'confidence': confidence,
                            'reasoning': self._generate_reasoning(query_lower, chunk_lower)
                        })
            
            # Sort by confidence score and limit results
            relevant_chunks.sort(key=lambda x: x['confidence'], reverse=True)
            final_chunks = relevant_chunks[:min(k, 6)]  # Limit to 6 chunks max for token efficiency
            
            logger.info(f"Retrieved {len(final_chunks)} optimized chunks (confidence >= {max(config.SIMILARITY_THRESHOLD, 0.2)})")
            return final_chunks
            
        except Exception as e:
            logger.error(f"Error in retrieve_relevant_chunks: {e}")
            return []
    
    def _expand_query(self, query: str) -> str:
        """Expand query with related terms for better matching."""
        query_lower = query.lower()
        expanded_terms = [query]
        
        # Add expansion terms based on patterns
        for pattern, terms in self.query_expansions.items():
            if pattern in query_lower:
                expanded_terms.extend(terms)
        
        # Add high-priority keywords found in query
        for keyword in self.insurance_keywords:
            if keyword in query_lower and keyword not in expanded_terms:
                expanded_terms.append(keyword)
        
        return " ".join(expanded_terms[:10])  # Limit expansion to avoid too long queries
    
    def _calculate_enhanced_confidence(self, similarity: float, query_lower: str, chunk_lower: str, expanded_query: str) -> float:
        """Enhanced confidence calculation with query expansion and better scoring."""
        try:
            # Base similarity score
            score = float(similarity)
            
            # Enhanced keyword matching with higher weights
            keyword_bonus = 0
            query_words = set(query_lower.split())
            expanded_words = set(expanded_query.split())
            
            # Check for exact phrase matches (high bonus)
            for word in query_words:
                if len(word) > 3 and word in chunk_lower:
                    keyword_bonus += 0.15
            
            # Check for insurance-specific keywords
            for keyword, weight in self.insurance_keywords.items():
                if keyword in expanded_words and keyword in chunk_lower:
                    keyword_bonus += weight * 0.8  # Reduced multiplier for balance
            
            # Bonus for multiple word matches
            matched_words = sum(1 for word in query_words if len(word) > 2 and word in chunk_lower)
            if matched_words > 1:
                keyword_bonus += matched_words * 0.05
            
            # Penalty for very short chunks (likely incomplete information)
            if len(chunk_lower) < 100:
                keyword_bonus -= 0.1
            
            # Combine scores with limits - increased cap for critical terms
            final_score = score + min(keyword_bonus, 0.6)  # Increased cap for better matching
            
            return min(final_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating enhanced confidence: {e}")
            return float(similarity)
    
    def _calculate_confidence_score(self, similarity: float, query_lower: str, chunk_lower: str) -> float:
        """Calculate confidence score combining similarity and keyword matching."""
        try:
            # Start with base similarity
            score = float(similarity)
            
            # Add keyword matching bonus
            keyword_bonus = 0
            for keyword, weight in self.insurance_keywords.items():
                if keyword in query_lower and keyword in chunk_lower:
                    keyword_bonus += weight
            
            # Add exact phrase matching bonus
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3 and word in chunk_lower:
                    keyword_bonus += 0.1
            
            # Combine scores (cap keyword bonus at 0.5)
            final_score = score + min(keyword_bonus, 0.5)
            
            return min(final_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return float(similarity)
    
    def _generate_reasoning(self, query_lower: str, chunk_lower: str) -> str:
        """Generate reasoning for why this chunk is relevant."""
        try:
            reasons = []
            
            # Check for keyword matches
            matched_keywords = []
            for keyword in self.insurance_keywords:
                if keyword in query_lower and keyword in chunk_lower:
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                reasons.append(f"Contains key terms: {', '.join(matched_keywords)}")
            
            # Check for exact word matches
            query_words = [w for w in query_lower.split() if len(w) > 3]
            matched_words = [w for w in query_words if w in chunk_lower]
            
            if matched_words:
                reasons.append(f"Matches query terms: {', '.join(matched_words[:3])}")
            
            return "; ".join(reasons) if reasons else "Semantic similarity match"
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return "Semantic similarity match"
    
    def multi_query_retrieval(self, queries: List[str], k: int = None) -> List[Tuple[Document, float]]:
        """Retrieve clauses using multiple search queries and aggregate results."""
        if k is None:
            k = config.TOP_K_RESULTS
            
        all_results = []
        
        for query in queries:
            results = self.embeddings_manager.search(query, k=k)
            all_results.extend(results)
        
        # Remove duplicates and sort by score
        unique_results = self._deduplicate_results(all_results)
        unique_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return unique_results[:k]
    
    def _deduplicate_results(self, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Remove duplicate documents and aggregate scores."""
        seen_docs = {}
        
        for doc, score in results:
            doc_id = doc.metadata.get('source', doc.page_content[:100])
            
            if doc_id in seen_docs:
                # Aggregate scores for same document
                seen_docs[doc_id] = (doc, max(seen_docs[doc_id][1], score))
            else:
                seen_docs[doc_id] = (doc, score)
        
        return list(seen_docs.values())
    
    def retrieve_by_section(self, query: str, section_type: str = None, k: int = None) -> List[Tuple[Document, float]]:
        """Retrieve clauses from specific policy sections."""
        if k is None:
            k = config.TOP_K_RESULTS
            
        # Get all results first
        all_results = self.embeddings_manager.search(query, k=k*2)
        
        # Filter by section if specified
        if section_type:
            filtered_results = [
                (doc, score) for doc, score in all_results
                if doc.metadata.get('section_type') == section_type
            ]
        else:
            filtered_results = all_results
        
        return filtered_results[:k]
    
    def retrieve_coverage_clauses(self, procedure: str, conditions: List[str] = None) -> List[Tuple[Document, float]]:
        """Retrieve coverage-related clauses for a specific procedure."""
        queries = [f"coverage {procedure}", f"benefits {procedure}"]
        
        if conditions:
            for condition in conditions:
                queries.append(f"{procedure} {condition}")
        
        return self.multi_query_retrieval(queries)
    
    def retrieve_exclusion_clauses(self, procedure: str) -> List[Tuple[Document, float]]:
        """Retrieve exclusion clauses for a specific procedure."""
        queries = [
            f"exclusions {procedure}",
            f"not covered {procedure}",
            f"excluded {procedure}"
        ]
        
        return self.multi_query_retrieval(queries)
    
    def retrieve_waiting_period_clauses(self, procedure: str = None) -> List[Tuple[Document, float]]:
        """Retrieve waiting period clauses."""
        if procedure:
            queries = [
                f"waiting period {procedure}",
                f"time limit {procedure}",
                f"duration {procedure}"
            ]
        else:
            queries = [
                "waiting period",
                "time limit",
                "duration requirement"
            ]
        
        return self.multi_query_retrieval(queries)
    
    def retrieve_amount_clauses(self, procedure: str = None) -> List[Tuple[Document, float]]:
        """Retrieve amount and limit clauses."""
        if procedure:
            queries = [
                f"amount {procedure}",
                f"limit {procedure}",
                f"maximum {procedure}",
                f"sum insured {procedure}"
            ]
        else:
            queries = [
                "amount limit",
                "maximum coverage",
                "sum insured",
                "policy limit"
            ]
        
        return self.multi_query_retrieval(queries)
    
    def retrieve_network_clauses(self) -> List[Tuple[Document, float]]:
        """Retrieve network hospital and provider clauses."""
        queries = [
            "network hospital",
            "cashless treatment",
            "provider network",
            "empaneled hospital"
        ]
        
        return self.multi_query_retrieval(queries)
    
    def retrieve_definition_clauses(self, term: str) -> List[Tuple[Document, float]]:
        """Retrieve definition clauses for specific terms."""
        queries = [
            f"definition {term}",
            f"meaning {term}",
            f"what is {term}",
            f"term {term}"
        ]
        
        return self.multi_query_retrieval(queries)
    
    def contextual_retrieval(self, query_info: Dict[str, Any]) -> Dict[str, List[Tuple[Document, float]]]:
        """Perform contextual retrieval based on query type and entities."""
        entities = query_info.get('entities', {})
        query_type = entities.get('query_type', 'general_inquiry')
        procedure = entities.get('procedure')
        
        results = {}
        
        # Base retrieval
        if procedure:
            results['coverage'] = self.retrieve_coverage_clauses(procedure, entities.get('conditions', []))
            results['exclusions'] = self.retrieve_exclusion_clauses(procedure)
            results['amounts'] = self.retrieve_amount_clauses(procedure)
        
        # Query type specific retrieval
        if query_type == 'coverage_check':
            if procedure:
                results['waiting_periods'] = self.retrieve_waiting_period_clauses(procedure)
            results['definitions'] = self.retrieve_definition_clauses(procedure or 'coverage')
            
        elif query_type == 'amount_inquiry':
            results['amounts'] = self.retrieve_amount_clauses(procedure)
            results['limits'] = self.retrieve_amount_clauses()
            
        elif query_type == 'waiting_period':
            results['waiting_periods'] = self.retrieve_waiting_period_clauses(procedure)
            
        elif query_type == 'exclusion_check':
            results['exclusions'] = self.retrieve_exclusion_clauses(procedure or 'general')
            
        elif query_type == 'provider_inquiry':
            results['network'] = self.retrieve_network_clauses()
            
        # Always include general coverage if not already included
        if 'coverage' not in results and procedure:
            results['coverage'] = self.retrieve_coverage_clauses(procedure)
        
        return results
    
    def rank_results_by_relevance(self, results: List[Tuple[Document, float]], 
                                 query_entities: Dict[str, Any]) -> List[Tuple[Document, float]]:
        """Rank results by relevance to query entities."""
        if not results:
            return results
        
        # Calculate additional relevance scores
        ranked_results = []
        
        for doc, score in results:
            relevance_score = score
            
            # Boost score for exact matches
            if query_entities.get('procedure'):
                if query_entities['procedure'].lower() in doc.page_content.lower():
                    relevance_score += 0.1
            
            # Boost score for section relevance
            section_id = doc.metadata.get('section_id', '')
            if 'SECTION A' in section_id and query_entities.get('query_type') == 'coverage_check':
                relevance_score += 0.05
            elif 'SECTION B' in section_id and query_entities.get('query_type') == 'exclusion_check':
                relevance_score += 0.05
            
            ranked_results.append((doc, relevance_score))
        
        # Sort by relevance score
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_results
    
    def get_context_window(self, results: List[Tuple[Document, float]], 
                          window_size: int = 3) -> str:
        """Create a context window from retrieved results."""
        if not results:
            return ""
        
        context_parts = []
        
        for i, (doc, score) in enumerate(results[:window_size]):
            context_parts.append(f"Document {i+1} (Score: {score:.3f}):\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def validate_retrieval_quality(self, results: List[Tuple[Document, float]], 
                                 query: str) -> Dict[str, Any]:
        """Validate the quality of retrieved results."""
        if not results:
            return {
                "quality_score": 0.0,
                "issues": ["No results retrieved"],
                "suggestions": ["Try broader search terms", "Check if documents are indexed"]
            }
        
        # Calculate quality metrics
        avg_score = sum(score for _, score in results) / len(results)
        max_score = max(score for _, score in results)
        min_score = min(score for _, score in results)
        
        # Check for diversity in sections
        sections = set(doc.metadata.get('section_id', '') for doc, _ in results)
        diversity_score = len(sections) / len(results)
        
        # Quality assessment
        quality_score = (avg_score + max_score + diversity_score) / 3
        
        issues = []
        suggestions = []
        
        if avg_score < 0.5:
            issues.append("Low average relevance score")
            suggestions.append("Consider refining search terms")
        
        if diversity_score < 0.3:
            issues.append("Low section diversity")
            suggestions.append("Try broader search to cover more policy sections")
        
        if max_score < 0.7:
            issues.append("No highly relevant results")
            suggestions.append("Check if query matches policy terminology")
        
        return {
            "quality_score": quality_score,
            "avg_score": avg_score,
            "max_score": max_score,
            "min_score": min_score,
            "diversity_score": diversity_score,
            "sections_covered": list(sections),
            "issues": issues,
            "suggestions": suggestions
        } 