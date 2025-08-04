import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
import json

from config import config

logger = logging.getLogger(__name__)

@dataclass
class DecisionResult:
    """Structured decision result."""
    decision: str
    confidence: float
    reasoning: str
    applicable_clauses: List[Dict[str, Any]]
    conditions: List[str]
    exclusions: List[str]
    amounts: Dict[str, Any]
    waiting_periods: List[str]

class DecisionEngine:
    """Evaluates policy logic and makes coverage decisions using an LLM-based approach."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        
    def evaluate_coverage(self, query: str, retrieved_clauses: List[Dict[str, Any]]) -> DecisionResult:
        """Main method to evaluate coverage based on query and retrieved clauses."""
        logger.info("Evaluating coverage decision using LLM-based reasoning.")
        
        if not retrieved_clauses:
            return DecisionResult(
                decision="no_information",
                confidence=0.0,
                reasoning="No relevant information was found in the policy document.",
                applicable_clauses=[],
                conditions=[],
                exclusions=[],
                amounts={},
                waiting_periods=[]
            )

        llm_result = self._refine_decision_with_llm(query, retrieved_clauses)
        
        return DecisionResult(
            decision=llm_result.get('decision', 'no_information'),
            confidence=llm_result.get('confidence', 0.5),
            reasoning=llm_result.get('reasoning', "LLM could not provide a reason."),
            applicable_clauses=retrieved_clauses,
            conditions=llm_result.get('conditions', []),
            exclusions=llm_result.get('exclusions', []),
            amounts=llm_result.get('amounts', {}),
            waiting_periods=llm_result.get('waiting_periods', [])
        )
    
    def _refine_decision_with_llm(self, query: str, retrieved_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM to refine the decision and provide a structured reasoning."""
        try:
            context_parts = [f"Clause {i+1}:\n{clause['content']}" for i, clause in enumerate(retrieved_clauses)]
            context = "\n\n".join(context_parts)
            
            prompt = f"""
            You are an expert insurance policy analyst. Your task is to analyze the provided policy clauses and determine the coverage status for a user's question. You must provide a structured JSON response.

            USER QUESTION: {query}
            
            POLICY CLAUSES:
            {context}
            
            INSTRUCTIONS:
            1.  Carefully read the question and all provided policy clauses.
            2.  Determine a final `decision` from the options: "covered", "not_covered", "partial_coverage", "conditional", or "no_information".
            3.  Provide a clear `reasoning` based *only* on the text in the policy clauses.
            4.  Extract all `conditions`, `exclusions`, `waiting_periods`, and specific `amounts` mentioned in the clauses related to the question. If a specific condition or amount is not found, leave the list or object empty.
            5.  Assign a `confidence` score from 0.0 (low) to 1.0 (high) based on how clearly and directly the clauses answer the question.
            
            Provide the response as a single JSON object.

            JSON Response:
            """
            
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            llm_output = response.choices[0].message.content
            
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                raise ValueError("LLM response did not contain a valid JSON object.")
            
            return result
            
        except Exception as e:
            logger.error(f"Error refining decision with LLM: {e}")
            return {
                'decision': 'no_information',
                'confidence': 0.0,
                'reasoning': f"Failed to analyze policy clauses: {str(e)}",
                'conditions': [],
                'exclusions': [],
                'amounts': {},
                'waiting_periods': []
            }