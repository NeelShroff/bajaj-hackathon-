import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI

from config import config

logger = logging.getLogger(__name__)

@dataclass
class DecisionResult:
    """Structured decision result."""
    decision: str  # "covered", "not_covered", "partial_coverage", "conditional"
    confidence: float
    reasoning: str
    applicable_clauses: List[Dict[str, Any]]
    conditions: List[str]
    exclusions_checked: List[str]
    amounts: Dict[str, Any]
    waiting_periods: List[str]

class DecisionEngine:
    """Evaluates policy logic and makes coverage decisions."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        
    def evaluate_coverage(self, query_info: Dict[str, Any], 
                         retrieved_clauses: Dict[str, List[Tuple[Any, float]]]) -> DecisionResult:
        """Main method to evaluate coverage based on query and retrieved clauses."""
        logger.info("Evaluating coverage decision")
        
        # Extract query entities
        entities = query_info.get('entities', {})
        procedure = entities.get('procedure')
        age = entities.get('age')
        conditions = entities.get('conditions', [])
        
        # Initialize decision components
        decision = "not_covered"
        confidence = 0.5
        reasoning = ""
        applicable_clauses = []
        conditions_list = []
        exclusions_checked = []
        amounts = {}
        waiting_periods = []
        
        # Check coverage clauses
        coverage_clauses = retrieved_clauses.get('coverage', [])
        if coverage_clauses:
            coverage_result = self._evaluate_coverage_clauses(coverage_clauses, entities)
            if coverage_result['covered']:
                decision = "covered"
                confidence = coverage_result['confidence']
                reasoning = coverage_result['reasoning']
                applicable_clauses.extend(coverage_result['clauses'])
                conditions_list.extend(coverage_result['conditions'])
                amounts.update(coverage_result.get('amounts', {}))
        
        # Check exclusions
        exclusion_clauses = retrieved_clauses.get('exclusions', [])
        if exclusion_clauses:
            exclusion_result = self._evaluate_exclusion_clauses(exclusion_clauses, entities)
            if exclusion_result['excluded']:
                decision = "not_covered"
                confidence = exclusion_result['confidence']
                reasoning = exclusion_result['reasoning']
                applicable_clauses.extend(exclusion_result['clauses'])
                exclusions_checked.extend(exclusion_result['exclusions'])
        
        # Check waiting periods
        waiting_clauses = retrieved_clauses.get('waiting_periods', [])
        if waiting_clauses:
            waiting_result = self._evaluate_waiting_periods(waiting_clauses, entities)
            if waiting_result['waiting_required']:
                decision = "conditional"
                confidence = min(confidence, waiting_result['confidence'])
                reasoning += f" {waiting_result['reasoning']}"
                waiting_periods.extend(waiting_result['periods'])
        
        # Use LLM for final decision refinement
        llm_result = self._refine_decision_with_llm(
            query_info, retrieved_clauses, decision, confidence, reasoning
        )
        
        return DecisionResult(
            decision=llm_result['decision'],
            confidence=llm_result['confidence'],
            reasoning=llm_result['reasoning'],
            applicable_clauses=applicable_clauses,
            conditions=conditions_list,
            exclusions_checked=exclusions_checked,
            amounts=amounts,
            waiting_periods=waiting_periods
        )
    
    def _evaluate_coverage_clauses(self, clauses: List[Tuple[Any, float]], 
                                 entities: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate coverage clauses for applicability."""
        covered = False
        confidence = 0.0
        reasoning = ""
        applicable_clauses = []
        conditions = []
        amounts = {}
        
        for doc, score in clauses:
            clause_text = doc.page_content.lower()
            
            # Check if procedure is mentioned in coverage
            if entities.get('procedure'):
                procedure_lower = entities['procedure'].lower()
                if procedure_lower in clause_text:
                    covered = True
                    confidence = max(confidence, score)
                    applicable_clauses.append({
                        'clause_id': doc.metadata.get('source', 'unknown'),
                        'section': doc.metadata.get('section_id', 'unknown'),
                        'text': doc.page_content,
                        'relevance_score': score
                    })
                    
                    # Extract conditions
                    conditions.extend(self._extract_conditions_from_clause(clause_text))
                    
                    # Extract amounts
                    amounts.update(self._extract_amounts_from_clause(clause_text))
        
        if covered:
            reasoning = f"Procedure '{entities.get('procedure')}' is covered under the policy."
        
        return {
            'covered': covered,
            'confidence': confidence,
            'reasoning': reasoning,
            'clauses': applicable_clauses,
            'conditions': conditions,
            'amounts': amounts
        }
    
    def _evaluate_exclusion_clauses(self, clauses: List[Tuple[Any, float]], 
                                  entities: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate exclusion clauses for applicability."""
        excluded = False
        confidence = 0.0
        reasoning = ""
        applicable_clauses = []
        exclusions = []
        
        for doc, score in clauses:
            clause_text = doc.page_content.lower()
            
            # Check if procedure is excluded
            if entities.get('procedure'):
                procedure_lower = entities['procedure'].lower()
                if procedure_lower in clause_text:
                    excluded = True
                    confidence = max(confidence, score)
                    applicable_clauses.append({
                        'clause_id': doc.metadata.get('source', 'unknown'),
                        'section': doc.metadata.get('section_id', 'unknown'),
                        'text': doc.page_content,
                        'relevance_score': score
                    })
                    
                    # Extract exclusion reasons
                    exclusions.extend(self._extract_exclusions_from_clause(clause_text))
        
        if excluded:
            reasoning = f"Procedure '{entities.get('procedure')}' is excluded from coverage."
        
        return {
            'excluded': excluded,
            'confidence': confidence,
            'reasoning': reasoning,
            'clauses': applicable_clauses,
            'exclusions': exclusions
        }
    
    def _evaluate_waiting_periods(self, clauses: List[Tuple[Any, float]], 
                                entities: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate waiting period requirements."""
        waiting_required = False
        confidence = 0.0
        reasoning = ""
        periods = []
        
        for doc, score in clauses:
            clause_text = doc.page_content.lower()
            
            # Extract waiting periods
            period_matches = re.findall(r'(\d+)\s*(day|month|year)s?', clause_text)
            for match in period_matches:
                duration = int(match[0])
                unit = match[1]
                periods.append(f"{duration} {unit}s")
                waiting_required = True
                confidence = max(confidence, score)
        
        if waiting_required:
            reasoning = f"Waiting period of {', '.join(periods)} applies."
        
        return {
            'waiting_required': waiting_required,
            'confidence': confidence,
            'reasoning': reasoning,
            'periods': periods
        }
    
    def _extract_conditions_from_clause(self, clause_text: str) -> List[str]:
        """Extract conditions from policy clause."""
        conditions = []
        
        # Common condition patterns
        condition_patterns = [
            r'provided\s+that\s+([^.]*)',
            r'if\s+([^.]*)',
            r'when\s+([^.]*)',
            r'subject\s+to\s+([^.]*)',
            r'condition\s+([^.]*)'
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, clause_text, re.IGNORECASE)
            for match in matches:
                conditions.append(match.strip())
        
        return conditions
    
    def _extract_amounts_from_clause(self, clause_text: str) -> Dict[str, Any]:
        """Extract monetary amounts from policy clause."""
        amounts = {}
        
        # Amount patterns
        amount_patterns = [
            (r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees?|inr|₹)', 'amount'),
            (r'maximum\s+(\d+(?:,\d+)*(?:\.\d+)?)', 'maximum'),
            (r'limit\s+of\s+(\d+(?:,\d+)*(?:\.\d+)?)', 'limit'),
            (r'sum\s+insured\s+(\d+(?:,\d+)*(?:\.\d+)?)', 'sum_insured')
        ]
        
        for pattern, amount_type in amount_patterns:
            matches = re.findall(pattern, clause_text, re.IGNORECASE)
            if matches:
                try:
                    amount = float(matches[0].replace(',', ''))
                    amounts[amount_type] = amount
                except ValueError:
                    continue
        
        return amounts
    
    def _extract_exclusions_from_clause(self, clause_text: str) -> List[str]:
        """Extract exclusion reasons from policy clause."""
        exclusions = []
        
        # Common exclusion patterns
        exclusion_patterns = [
            r'not\s+covered\s+([^.]*)',
            r'excluded\s+([^.]*)',
            r'exclusion\s+([^.]*)',
            r'not\s+eligible\s+([^.]*)'
        ]
        
        for pattern in exclusion_patterns:
            matches = re.findall(pattern, clause_text, re.IGNORECASE)
            for match in matches:
                exclusions.append(match.strip())
        
        return exclusions
    
    def _refine_decision_with_llm(self, query_info: Dict[str, Any], 
                                retrieved_clauses: Dict[str, List[Tuple[Any, float]]],
                                initial_decision: str, confidence: float, 
                                reasoning: str) -> Dict[str, Any]:
        """Use LLM to refine the decision and reasoning."""
        try:
            # Prepare context from retrieved clauses
            context_parts = []
            for category, clauses in retrieved_clauses.items():
                if clauses:
                    context_parts.append(f"{category.upper()} CLAUSES:")
                    for i, (doc, score) in enumerate(clauses[:3]):  # Top 3 clauses
                        context_parts.append(f"{i+1}. {doc.page_content}")
                    context_parts.append("")
            
            context = "\n".join(context_parts)
            
            prompt = f"""
            Based on the following insurance policy query and retrieved clauses, provide a final coverage decision:
            
            QUERY: {query_info.get('original_query', '')}
            QUERY ENTITIES: {query_info.get('entities', {})}
            
            RETRIEVED CLAUSES:
            {context}
            
            INITIAL DECISION: {initial_decision}
            CONFIDENCE: {confidence}
            REASONING: {reasoning}
            
            Please provide a JSON response with:
            - decision: "covered", "not_covered", "partial_coverage", or "conditional"
            - confidence: confidence score (0-1)
            - reasoning: clear explanation of the decision
            - key_factors: list of key factors that influenced the decision
            
            Return only the JSON object.
            """
            
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return {
                'decision': result.get('decision', initial_decision),
                'confidence': result.get('confidence', confidence),
                'reasoning': result.get('reasoning', reasoning),
                'key_factors': result.get('key_factors', [])
            }
            
        except Exception as e:
            logger.error(f"Error refining decision with LLM: {e}")
            return {
                'decision': initial_decision,
                'confidence': confidence,
                'reasoning': reasoning,
                'key_factors': []
            }
    
    def calculate_coverage_amount(self, decision_result: DecisionResult, 
                                query_entities: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate coverage amounts based on decision and policy limits."""
        amounts = decision_result.amounts.copy()
        
        # Default amounts if not specified
        if not amounts:
            amounts = {
                'amount': 50000,  # Default coverage amount
                'currency': 'INR'
            }
        
        # Calculate patient responsibility (typically 10-20% of covered amount)
        covered_amount = amounts.get('amount', 50000)
        patient_responsibility = covered_amount * 0.1  # 10% co-pay
        
        amounts.update({
            'covered_amount': covered_amount,
            'patient_responsibility': patient_responsibility,
            'currency': amounts.get('currency', 'INR')
        })
        
        return amounts
    
    def validate_decision(self, decision_result: DecisionResult, 
                         query_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the decision quality and completeness."""
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check decision confidence
        if decision_result.confidence < 0.5:
            validation['warnings'].append("Low confidence decision")
            validation['suggestions'].append("Consider manual review")
        
        # Check reasoning completeness
        if len(decision_result.reasoning) < 50:
            validation['issues'].append("Insufficient reasoning provided")
            validation['suggestions'].append("Enhance decision explanation")
        
        # Check applicable clauses
        if not decision_result.applicable_clauses:
            validation['warnings'].append("No applicable clauses identified")
            validation['suggestions'].append("Review clause retrieval")
        
        # Check for contradictions
        if decision_result.decision == "covered" and decision_result.exclusions_checked:
            validation['warnings'].append("Coverage granted despite exclusions found")
        
        if validation['issues']:
            validation['is_valid'] = False
        
        return validation 