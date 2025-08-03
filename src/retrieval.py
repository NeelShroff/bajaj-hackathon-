import logging
from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document

from .embeddings import EmbeddingsManager
from config import config

logger = logging.getLogger(__name__)

class RetrievalSystem:
    """Handles semantic search and retrieval of relevant policy clauses."""
    
    def __init__(self, embeddings_manager: EmbeddingsManager):
        self.embeddings_manager = embeddings_manager
        
    def retrieve_relevant_clauses(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Retrieve relevant policy clauses for a given query."""
        if k is None:
            k = config.TOP_K_RESULTS
            
        logger.info(f"Retrieving relevant clauses for query: {query}")
        
        # Perform semantic search
        results = self.embeddings_manager.search_with_threshold(query, k=k)
        
        logger.info(f"Retrieved {len(results)} relevant clauses")
        return results
    
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