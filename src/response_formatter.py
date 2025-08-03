import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from .decision_engine import DecisionResult

logger = logging.getLogger(__name__)

class ResponseFormatter:
    """Formats decision results into structured JSON responses."""
    
    def __init__(self):
        self.start_time = None
        
    def start_timing(self):
        """Start timing the response generation."""
        self.start_time = time.time()
    
    def format_response(self, query: str, query_info: Dict[str, Any], 
                       decision_result: DecisionResult, 
                       retrieved_clauses: Dict[str, List[tuple]],
                       document_source: str = None) -> Dict[str, Any]:
        """Format the complete response with all components."""
        self.start_timing()
        
        # Calculate processing time
        processing_time = time.time() - self.start_time if self.start_time else 0
        
        # Format amounts
        amounts = self._format_amounts(decision_result.amounts)
        
        # Format justification
        justification = self._format_justification(decision_result, retrieved_clauses)
        
        # Format metadata
        metadata = self._format_metadata(processing_time, document_source, query_info)
        
        # Build response
        response = {
            "query": query,
            "decision": decision_result.decision,
            "confidence": round(decision_result.confidence, 3),
            "amount": amounts,
            "justification": justification,
            "metadata": metadata
        }
        
        # Add query analysis if available
        if query_info.get('entities'):
            response["query_analysis"] = {
                "extracted_entities": query_info['entities'],
                "search_queries": query_info.get('search_queries', []),
                "processing_confidence": query_info.get('processing_metadata', {}).get('confidence', 0.0)
            }
        
        return response
    
    def _format_amounts(self, amounts: Dict[str, Any]) -> Dict[str, Any]:
        """Format amount information."""
        if not amounts:
            return {
                "covered_amount": 0,
                "patient_responsibility": 0,
                "currency": "INR"
            }
        
        # Ensure all required fields are present
        formatted_amounts = {
            "covered_amount": amounts.get('covered_amount', amounts.get('amount', 0)),
            "patient_responsibility": amounts.get('patient_responsibility', 0),
            "currency": amounts.get('currency', 'INR')
        }
        
        # Add additional amount fields if available
        if 'maximum' in amounts:
            formatted_amounts['maximum_limit'] = amounts['maximum']
        if 'sum_insured' in amounts:
            formatted_amounts['sum_insured'] = amounts['sum_insured']
        if 'limit' in amounts:
            formatted_amounts['policy_limit'] = amounts['limit']
        
        return formatted_amounts
    
    def _format_justification(self, decision_result: DecisionResult, 
                            retrieved_clauses: Dict[str, List[tuple]]) -> Dict[str, Any]:
        """Format the justification section."""
        justification = {
            "reasoning": decision_result.reasoning,
            "applicable_clauses": self._format_applicable_clauses(decision_result.applicable_clauses),
            "conditions": decision_result.conditions,
            "exclusions_checked": decision_result.exclusions_checked
        }
        
        # Add waiting periods if applicable
        if decision_result.waiting_periods:
            justification["waiting_periods"] = decision_result.waiting_periods
        
        # Add retrieval quality information
        retrieval_quality = self._assess_retrieval_quality(retrieved_clauses)
        if retrieval_quality:
            justification["retrieval_quality"] = retrieval_quality
        
        return justification
    
    def _format_applicable_clauses(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format applicable clauses for response."""
        formatted_clauses = []
        
        for clause in clauses:
            formatted_clause = {
                "clause_id": clause.get('clause_id', 'unknown'),
                "section": clause.get('section', 'unknown'),
                "text": clause.get('text', ''),
                "relevance_score": round(clause.get('relevance_score', 0), 3)
            }
            formatted_clauses.append(formatted_clause)
        
        return formatted_clauses
    
    def _assess_retrieval_quality(self, retrieved_clauses: Dict[str, List[tuple]]) -> Dict[str, Any]:
        """Assess the quality of retrieved clauses."""
        total_clauses = sum(len(clauses) for clauses in retrieved_clauses.values())
        
        if total_clauses == 0:
            return {
                "quality_score": 0.0,
                "issues": ["No clauses retrieved"],
                "suggestions": ["Check document indexing", "Verify query relevance"]
            }
        
        # Calculate average relevance scores
        all_scores = []
        for clauses in retrieved_clauses.values():
            all_scores.extend([score for _, score in clauses])
        
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        max_score = max(all_scores) if all_scores else 0
        
        # Assess quality
        quality_score = (avg_score + max_score) / 2
        
        issues = []
        suggestions = []
        
        if avg_score < 0.5:
            issues.append("Low average relevance")
            suggestions.append("Consider refining search terms")
        
        if max_score < 0.7:
            issues.append("No highly relevant results")
            suggestions.append("Check policy document coverage")
        
        return {
            "quality_score": round(quality_score, 3),
            "average_relevance": round(avg_score, 3),
            "max_relevance": round(max_score, 3),
            "total_clauses_retrieved": total_clauses,
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _format_metadata(self, processing_time: float, document_source: str, 
                        query_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format metadata information."""
        metadata = {
            "processing_time": f"{processing_time:.1f}s",
            "document_source": document_source or "unknown"
        }
        
        # Add token usage if available
        if query_info.get('processing_metadata', {}).get('tokens_used'):
            metadata["tokens_used"] = query_info['processing_metadata']['tokens_used']
        
        # Add processing method
        metadata["processing_method"] = query_info.get('processing_metadata', {}).get('method', 'hybrid')
        
        return metadata
    
    def format_error_response(self, query: str, error: str, error_type: str = "processing_error") -> Dict[str, Any]:
        """Format error response."""
        return {
            "query": query,
            "error": {
                "type": error_type,
                "message": error,
                "timestamp": time.time()
            },
            "decision": "error",
            "confidence": 0.0,
            "amount": {
                "covered_amount": 0,
                "patient_responsibility": 0,
                "currency": "INR"
            },
            "justification": {
                "reasoning": f"Unable to process query due to {error_type}",
                "applicable_clauses": [],
                "conditions": [],
                "exclusions_checked": []
            },
            "metadata": {
                "processing_time": f"{time.time() - self.start_time:.1f}s" if self.start_time else "unknown",
                "document_source": "unknown",
                "error_occurred": True
            }
        }
    
    def format_partial_response(self, query: str, partial_data: Dict[str, Any], 
                              missing_components: List[str]) -> Dict[str, Any]:
        """Format partial response when some components are missing."""
        response = {
            "query": query,
            "decision": partial_data.get('decision', 'partial'),
            "confidence": partial_data.get('confidence', 0.0),
            "amount": partial_data.get('amount', {
                "covered_amount": 0,
                "patient_responsibility": 0,
                "currency": "INR"
            }),
            "justification": partial_data.get('justification', {
                "reasoning": "Partial analysis completed",
                "applicable_clauses": [],
                "conditions": [],
                "exclusions_checked": []
            }),
            "metadata": {
                "processing_time": f"{time.time() - self.start_time:.1f}s" if self.start_time else "unknown",
                "document_source": partial_data.get('document_source', 'unknown'),
                "partial_response": True,
                "missing_components": missing_components
            }
        }
        
        return response
    
    def format_batch_response(self, queries: List[str], results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format response for batch processing."""
        return {
            "batch_processing": True,
            "total_queries": len(queries),
            "successful_queries": len([r for r in results if r.get('decision') != 'error']),
            "failed_queries": len([r for r in results if r.get('decision') == 'error']),
            "results": results,
            "metadata": {
                "processing_time": f"{time.time() - self.start_time:.1f}s" if self.start_time else "unknown",
                "average_confidence": sum(r.get('confidence', 0) for r in results) / len(results) if results else 0
            }
        }
    
    def format_health_response(self, system_status: Dict[str, Any]) -> Dict[str, Any]:
        """Format health check response."""
        return {
            "status": "healthy" if system_status.get('is_healthy', False) else "unhealthy",
            "timestamp": time.time(),
            "components": {
                "embeddings_manager": system_status.get('embeddings_healthy', False),
                "document_loader": system_status.get('document_loader_healthy', False),
                "openai_api": system_status.get('openai_healthy', False),
                "faiss_index": system_status.get('faiss_healthy', False)
            },
            "index_stats": system_status.get('index_stats', {}),
            "document_count": system_status.get('document_count', 0),
            "version": "1.0.0"
        } 