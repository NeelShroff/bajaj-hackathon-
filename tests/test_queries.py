"""
Test queries for the LLM Document Processing System.
These queries cover various scenarios and edge cases for testing.
"""

# Sample test queries for insurance policy analysis
test_queries = [
    # Basic coverage queries
    "46M, knee surgery, Pune, 3-month policy",
    "Does this policy cover maternity expenses?",
    "What is the waiting period for pre-existing diseases?",
    "Dental treatment coverage for emergency accident",
    "Room rent limits for ICU admission",
    "Coverage for AYUSH treatments",
    "Pre-hospitalization expenses duration",
    
    # Amount and limit queries
    "What is the maximum coverage amount for hospitalization?",
    "Sum insured limit for critical illness",
    "Room rent capping for private hospitals",
    "Maximum coverage for day care procedures",
    "Policy limit for outpatient treatment",
    
    # Exclusion queries
    "Are cosmetic surgeries covered?",
    "Exclusions for pre-existing conditions",
    "What treatments are not covered?",
    "Coverage for experimental treatments",
    "Exclusions for dental procedures",
    
    # Waiting period queries
    "Waiting period for maternity benefits",
    "How long to wait for pre-existing disease coverage?",
    "Time limit for critical illness coverage",
    "Waiting period for specific procedures",
    
    # Network and provider queries
    "Cashless treatment at network hospitals",
    "Reimbursement for non-network hospitals",
    "List of empaneled hospitals",
    "Provider network coverage",
    
    # Specific procedure queries
    "Coverage for heart bypass surgery",
    "Knee replacement surgery benefits",
    "Cancer treatment coverage",
    "Organ transplant coverage",
    "Mental health treatment coverage",
    
    # Age and demographic queries
    "Coverage for 65-year-old patient",
    "Policy benefits for senior citizens",
    "Age limit for dependent coverage",
    "Coverage for newborn babies",
    
    # Emergency and accident queries
    "Emergency accident coverage",
    "Road accident treatment benefits",
    "Emergency hospitalization coverage",
    "Accident-related dental treatment",
    
    # Complex scenario queries
    "46M male, knee surgery, emergency, Pune, 1-year policy, network hospital",
    "35F, maternity, pre-existing diabetes, Mumbai, 2-year policy",
    "28M, dental emergency, accident, Bangalore, 6-month policy",
    "55F, heart surgery, pre-existing condition, Delhi, 3-year policy",
    
    # Edge cases and vague queries
    "What's covered?",
    "Policy benefits",
    "How much will I get?",
    "Is it covered?",
    "Treatment coverage",
    "Medical expenses",
    
    # Specific condition queries
    "Diabetes treatment coverage",
    "Hypertension medication coverage",
    "Asthma treatment benefits",
    "Cancer chemotherapy coverage",
    "Dialysis treatment coverage",
    
    # Duration and timing queries
    "Pre-hospitalization period",
    "Post-hospitalization coverage duration",
    "Day care procedure time limit",
    "ICU stay duration coverage",
    
    # Location-specific queries
    "Treatment coverage in Mumbai",
    "Hospitalization benefits in Delhi",
    "Coverage for treatment in Bangalore",
    "Regional hospital coverage",
    
    # Policy type queries
    "Individual policy coverage",
    "Family floater benefits",
    "Group policy coverage",
    "Corporate policy benefits"
]

# Query categories for organized testing
query_categories = {
    "coverage_check": [
        "46M, knee surgery, Pune, 3-month policy",
        "Does this policy cover maternity expenses?",
        "Coverage for AYUSH treatments",
        "Are cosmetic surgeries covered?"
    ],
    
    "amount_inquiry": [
        "What is the maximum coverage amount for hospitalization?",
        "Sum insured limit for critical illness",
        "Room rent capping for private hospitals",
        "Maximum coverage for day care procedures"
    ],
    
    "waiting_period": [
        "What is the waiting period for pre-existing diseases?",
        "Waiting period for maternity benefits",
        "How long to wait for pre-existing disease coverage?",
        "Time limit for critical illness coverage"
    ],
    
    "exclusion_check": [
        "Exclusions for pre-existing conditions",
        "What treatments are not covered?",
        "Coverage for experimental treatments",
        "Exclusions for dental procedures"
    ],
    
    "provider_inquiry": [
        "Cashless treatment at network hospitals",
        "Reimbursement for non-network hospitals",
        "List of empaneled hospitals",
        "Provider network coverage"
    ],
    
    "general_inquiry": [
        "What's covered?",
        "Policy benefits",
        "How much will I get?",
        "Treatment coverage"
    ]
}

# Complex test scenarios
complex_scenarios = [
    {
        "scenario": "Emergency knee surgery for middle-aged male",
        "query": "46M male, knee surgery, emergency, Pune, 1-year policy, network hospital",
        "expected_entities": {
            "age": 46,
            "gender": "male",
            "procedure": "knee surgery",
            "location": "Pune",
            "policy_duration": "1 year",
            "conditions": ["emergency", "network"],
            "query_type": "coverage_check"
        }
    },
    
    {
        "scenario": "Maternity coverage with pre-existing condition",
        "query": "35F, maternity, pre-existing diabetes, Mumbai, 2-year policy",
        "expected_entities": {
            "age": 35,
            "gender": "female",
            "procedure": "maternity",
            "location": "Mumbai",
            "policy_duration": "2 years",
            "conditions": ["pre-existing"],
            "query_type": "coverage_check"
        }
    },
    
    {
        "scenario": "Dental emergency from accident",
        "query": "28M, dental emergency, accident, Bangalore, 6-month policy",
        "expected_entities": {
            "age": 28,
            "gender": "male",
            "procedure": "dental emergency",
            "location": "Bangalore",
            "policy_duration": "6 months",
            "conditions": ["accident"],
            "query_type": "coverage_check"
        }
    }
]

# Performance test queries (for stress testing)
performance_queries = [
    "Basic coverage query",
    "Complex multi-entity query",
    "Vague general query",
    "Specific procedure query",
    "Amount inquiry query",
    "Exclusion check query",
    "Waiting period query",
    "Provider inquiry query"
] * 10  # Repeat each query 10 times for performance testing

def get_test_queries_by_category(category: str = None) -> list:
    """Get test queries filtered by category."""
    if category is None:
        return test_queries
    
    return query_categories.get(category, [])

def get_complex_scenarios() -> list:
    """Get complex test scenarios."""
    return complex_scenarios

def get_performance_queries() -> list:
    """Get queries for performance testing."""
    return performance_queries

def get_random_queries(count: int = 5) -> list:
    """Get a random selection of test queries."""
    import random
    return random.sample(test_queries, min(count, len(test_queries)))

if __name__ == "__main__":
    # Print available test queries
    print("Available test queries:")
    for i, query in enumerate(test_queries[:10], 1):
        print(f"{i}. {query}")
    
    print(f"\nTotal test queries: {len(test_queries)}")
    print(f"Query categories: {list(query_categories.keys())}")
    print(f"Complex scenarios: {len(complex_scenarios)}") 