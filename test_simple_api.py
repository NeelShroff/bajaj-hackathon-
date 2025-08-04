#!/usr/bin/env python3
"""
Simple API test to debug the batch endpoint
"""

import requests
import json
import time

# Test data - All 10 comprehensive questions
test_data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

def test_simple_api():
    print("🧪 Testing Simple API (10 questions)")
    print("=" * 50)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token-12345"
    }
    
    start_time = time.time()
    
    try:
        print("🔄 Sending request...")
        response = requests.post(
            "http://localhost:8000/hackrx/run", 
            json=test_data, 
            headers=headers, 
            timeout=60
        )
        
        end_time = time.time()
        print(f"✅ Response in {end_time - start_time:.2f}s")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Answers: {len(result.get('answers', []))}")
            for i, answer in enumerate(result.get('answers', []), 1):
                print(f"{i}. {answer}")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Timeout after 60s")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_simple_api()
