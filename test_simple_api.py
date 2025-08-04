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
        "What is the maximum limit for domiciliary hospitalization for Plan C?",
        "What are the conditions for coverage of Morbid Obesity Treatment?",
        "How many days of post-hospitalisation expenses are covered?",
        "What are the eligibility criteria for surgical treatment of obesity?",
        "What is the time limit for submitting documents for a reimbursement claim after discharge from the hospital?",
        "Does the policy cover expenses for treatment for correction of eye sight due to refractive error?",
        "If a claim is found to be fraudulent, what happens to the benefits under the policy?",
        "Is there a waiting period for treatment of hypertension?",
        "What is the limit of cover for robotic surgery under Modern Treatments?",
        "What happens to the policy premium if the eldest insured person dies?"
    ]
}

def test_simple_api():
    print("🧪 Testing Simple API (10 questions)")
    print("=" * 50)
    
    # IMPORTANT: Update this token to match the secret token in your main.py server
    API_SECRET_TOKEN = "YOUR_SECURE_TOKEN_HERE"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_SECRET_TOKEN}"
    }
    
    start_time = time.time()
    
    try:
        print("🔄 Sending request to http://localhost:8000/hackrx/run...")
        response = requests.post(
            "http://localhost:8000/hackrx/run", 
            json=test_data, 
            headers=headers, 
            timeout=120
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            print(f"✅ Response received in {end_time - start_time:.2f}s")
            print(f"Status: {response.status_code}")
            result = response.json()
            print(f"Total answers: {len(result.get('answers', []))}")
            print(f"Total processing time: {result.get('processing_time', 'N/A'):.2f}s")
            print("-" * 20)
            
            for i, answer in enumerate(result.get('answers', []), 1):
                print(f"Question {i}: {test_data['questions'][i-1]}")
                print(f"Answer {i}: {answer}")
                print("-" * 20)
        else:
            print(f"❌ Error: Received status code {response.status_code}")
            print(f"Response body: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Timeout after 120s")
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("Please ensure the FastAPI server is running at http://localhost:8000")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_simple_api()