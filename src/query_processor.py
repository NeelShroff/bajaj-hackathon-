import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from openai import OpenAI

from config import config

logger = logging.getLogger(__name__)

@dataclass
class QueryEntities:
    """Structured representation of extracted query entities."""
    age: Optional[int] = None
    gender: Optional[str] = None
    procedure: Optional[str] = None
    location: Optional[str] = None
    policy_duration: Optional[str] = None
    query_type: Optional[str] = None
    conditions: List[str] = None
    amounts: List[float] = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []
        if self.amounts is None:
            self.amounts = []

class QueryProcessor:
    """Processes natural language queries and extracts structured information."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Common patterns for entity extraction
        self.age_pattern = r'(\d{1,3})(?:M|F|Male|Female|MALE|FEMALE)'
        self.gender_pattern = r'(\d{1,3})(M|F|Male|Female|MALE|FEMALE)'
        self.location_pattern = r'\b(Mumbai|Delhi|Bangalore|Chennai|Kolkata|Pune|Hyderabad|Ahmedabad|Jaipur|Lucknow|Kanpur|Nagpur|Indore|Thane|Bhopal|Visakhapatnam|Pimpri-Chinchwad|Patna|Vadodara|Ghaziabad|Ludhiana|Agra|Nashik|Faridabad|Meerut|Rajkot|Kalyan-Dombivali|Vasai-Virar|Varanasi|Srinagar|Aurangabad|Dhanbad|Amritsar|Allahabad|Ranchi|Howrah|Coimbatore|Jabalpur|Gwalior|Vijayawada|Jodhpur|Madurai|Raipur|Kota|Guwahati|Chandigarh|Solapur|Tiruchirappalli|Bareilly|Moradabad|Mysore|Tiruppur|Gurgaon|Aligarh|Jalandhar|Bhubaneswar|Salem|Warangal|Guntur|Bhiwandi|Saharanpur|Gorakhpur|Bikaner|Amravati|Noida|Jamshedpur|Bhilai|Cuttack|Firozabad|Kochi|Nellore|Bhavnagar|Dehradun|Durgapur|Asansol|Rourkela|Nanded|Kolhapur|Ajmer|Akola|Gulbarga|Jamnagar|Ujjain|Loni|Siliguri|Jhansi|Ulhasnagar|Jammu|Sangli-Miraj|Mangalore|Erode|Belgaum|Ambattur|Tirunelveli|Malegaon|Gaya|Jalgaon|Udaipur|Maheshtala|Tirupur|Davanagere|Kozhikode|Kurnool|Rajpur|Sonarpur|Bokaro|South|Dum|Dum|Durg|Raj|Nagar|Bihar|Sharif|Panihati|Satara|Bijapur|Brahmapur|Shahjahanpur|Bidar|Gandhidham|Baranagar|Tiruvottiyur|Puducherry|Sikar|Thrissur|Alwar|Bahraich|Phusro|Vellore|Mehsana|Raebareli|Chittoor|Gwalior|Bhilwara|Gandhinagar|Bharatpur|Sikar|Panipat|Fatehpur|Budhana|Okara|Sanand|Tonk|Gangtok|Faizabad|Muktsar|Khanna|Yavatmal|Dhule|Korba|Bokaro|Steel|City|Raj|Nagar|Bihar|Sharif|Panihati|Satara|Bijapur|Brahmapur|Shahjahanpur|Bidar|Gandhidham|Baranagar|Tiruvottiyur|Puducherry|Sikar|Thrissur|Alwar|Bahraich|Phusro|Vellore|Mehsana|Raebareli|Chittoor|Gwalior|Bhilwara|Gandhinagar|Bharatpur|Sikar|Panipat|Fatehpur|Budhana|Okara|Sanand|Tonk|Gangtok|Faizabad|Muktsar|Khanna|Yavatmal|Dhule|Korba|Bokaro|Steel|City)\b'
        self.duration_pattern = r'(\d+)\s*(month|year|day)s?'
        self.amount_pattern = r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees?|inr|₹)?'
        
    def extract_entities(self, query: str) -> QueryEntities:
        """Extract structured entities from natural language query."""
        query_lower = query.lower()
        entities = QueryEntities()
        
        # Extract age and gender
        age_gender_match = re.search(self.gender_pattern, query, re.IGNORECASE)
        if age_gender_match:
            entities.age = int(age_gender_match.group(1))
            gender = age_gender_match.group(2).upper()
            entities.gender = 'male' if gender in ['M', 'MALE'] else 'female'
        
        # Extract location
        location_match = re.search(self.location_pattern, query, re.IGNORECASE)
        if location_match:
            entities.location = location_match.group(1)
        
        # Extract policy duration
        duration_match = re.search(self.duration_pattern, query_lower)
        if duration_match:
            entities.policy_duration = f"{duration_match.group(1)} {duration_match.group(2)}s"
        
        # Extract amounts
        amount_matches = re.findall(self.amount_pattern, query, re.IGNORECASE)
        for amount_str in amount_matches:
            try:
                amount = float(amount_str.replace(',', ''))
                entities.amounts.append(amount)
            except ValueError:
                continue
        
        # Extract procedure/treatment
        entities.procedure = self._extract_procedure(query)
        
        # Determine query type
        entities.query_type = self._determine_query_type(query)
        
        # Extract conditions
        entities.conditions = self._extract_conditions(query)
        
        return entities
    
    def _extract_procedure(self, query: str) -> Optional[str]:
        """Extract medical procedure or treatment from query."""
        # Common medical procedures
        procedures = [
            'knee surgery', 'heart surgery', 'dental treatment', 'maternity',
            'hospitalization', 'emergency', 'accident', 'pre-existing',
            'disease', 'illness', 'surgery', 'treatment', 'therapy',
            'medication', 'prescription', 'consultation', 'diagnosis',
            'x-ray', 'mri', 'ct scan', 'ultrasound', 'blood test',
            'vaccination', 'immunization', 'physiotherapy', 'occupational therapy'
        ]
        
        query_lower = query.lower()
        for procedure in procedures:
            if procedure in query_lower:
                return procedure
        
        return None
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query being asked."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['cover', 'coverage', 'covered']):
            return 'coverage_check'
        elif any(word in query_lower for word in ['amount', 'limit', 'maximum', 'sum']):
            return 'amount_inquiry'
        elif any(word in query_lower for word in ['waiting', 'period', 'time']):
            return 'waiting_period'
        elif any(word in query_lower for word in ['exclude', 'exclusion', 'not cover']):
            return 'exclusion_check'
        elif any(word in query_lower for word in ['network', 'hospital', 'provider']):
            return 'provider_inquiry'
        else:
            return 'general_inquiry'
    
    def _extract_conditions(self, query: str) -> List[str]:
        """Extract conditions or modifiers from query."""
        conditions = []
        query_lower = query.lower()
        
        # Common conditions
        condition_keywords = [
            'emergency', 'accident', 'pre-existing', 'chronic',
            'acute', 'elective', 'planned', 'unplanned',
            'network', 'non-network', 'cashless', 'reimbursement',
            'day care', 'in-patient', 'out-patient', 'domiciliary'
        ]
        
        for condition in condition_keywords:
            if condition in query_lower:
                conditions.append(condition)
        
        return conditions
    
    def enhance_query_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM to enhance query understanding and extraction."""
        try:
            prompt = f"""
            Analyze the following insurance policy query and extract structured information:
            
            Query: "{query}"
            
            Please extract and return a JSON object with the following fields:
            - age: numeric age if mentioned
            - gender: "male", "female", or null
            - procedure: medical procedure/treatment mentioned
            - location: city/location if mentioned
            - policy_duration: policy duration if mentioned
            - query_type: "coverage_check", "amount_inquiry", "waiting_period", "exclusion_check", "provider_inquiry", or "general_inquiry"
            - conditions: list of conditions/modifiers (emergency, pre-existing, network, etc.)
            - amounts: list of monetary amounts mentioned
            - confidence: confidence score (0-1) for the extraction
            
            Return only the JSON object, no additional text.
            """
            
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse JSON response
            import json
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing query with LLM: {e}")
            # Fallback to rule-based extraction
            entities = self.extract_entities(query)
            return {
                "age": entities.age,
                "gender": entities.gender,
                "procedure": entities.procedure,
                "location": entities.location,
                "policy_duration": entities.policy_duration,
                "query_type": entities.query_type,
                "conditions": entities.conditions,
                "amounts": entities.amounts,
                "confidence": 0.7
            }
    
    def create_search_queries(self, entities: QueryEntities) -> List[str]:
        """Create multiple search queries for better retrieval."""
        queries = []
        
        # Base query with all entities
        base_query_parts = []
        if entities.procedure:
            base_query_parts.append(entities.procedure)
        if entities.age:
            base_query_parts.append(f"{entities.age} year old")
        if entities.gender:
            base_query_parts.append(entities.gender)
        if entities.location:
            base_query_parts.append(entities.location)
        
        if base_query_parts:
            queries.append(" ".join(base_query_parts))
        
        # Specific coverage query
        if entities.procedure:
            queries.append(f"coverage {entities.procedure}")
            queries.append(f"benefits {entities.procedure}")
        
        # Exclusion query
        if entities.procedure:
            queries.append(f"exclusions {entities.procedure}")
        
        # Waiting period query
        if entities.procedure:
            queries.append(f"waiting period {entities.procedure}")
        
        # Amount/limit query
        if entities.procedure:
            queries.append(f"amount limit {entities.procedure}")
            queries.append(f"maximum coverage {entities.procedure}")
        
        return queries
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Main method to process a query and return structured information."""
        logger.info(f"Processing query: {query}")
        
        # Extract entities using rule-based approach
        entities = self.extract_entities(query)
        
        # Enhance with LLM if needed
        enhanced_result = self.enhance_query_with_llm(query)
        
        # Create search queries
        search_queries = self.create_search_queries(entities)
        
        return {
            "original_query": query,
            "entities": {
                "age": entities.age,
                "gender": entities.gender,
                "procedure": entities.procedure,
                "location": entities.location,
                "policy_duration": entities.policy_duration,
                "query_type": entities.query_type,
                "conditions": entities.conditions,
                "amounts": entities.amounts
            },
            "enhanced_extraction": enhanced_result,
            "search_queries": search_queries,
            "processing_metadata": {
                "method": "hybrid",
                "confidence": enhanced_result.get("confidence", 0.7)
            }
        } 