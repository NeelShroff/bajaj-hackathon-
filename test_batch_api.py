#!/usr/bin/env python3
"""
Test script for batch API endpoint to measure performance and token usage
"""

import requests
import json
import time
from datetime import datetime

# API endpoint
API_URL = "http://localhost:8000/hackrx/run"

# Test data with the hackathon sample questions
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

# Expected answers for comparison
expected_answers = [
    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
    "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
    "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
    "The policy has a specific waiting period of two (2) years for cataract surgery.",
    "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
    "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
    "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
    "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
    "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
    "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
]

def test_batch_api():
    """Test the batch API endpoint and measure performance."""
    print("🚀 Testing Batch API Endpoint")
    print("=" * 60)
    
    # Headers with Bearer token (required for authentication)
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token-12345"  # Simple test token
    }
    
    print(f"📊 Testing with {len(test_data['questions'])} questions")
    print(f"📄 Document: {test_data['documents'][:50]}...")
    print(f"⏰ Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Measure total time
    start_time = time.time()
    
    try:
        # Make the API request
        print("\n🔄 Sending batch request...")
        response = requests.post(API_URL, json=test_data, headers=headers, timeout=120)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Response received in {total_time:.2f} seconds")
        print(f"📈 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answers = result.get("answers", [])
            metadata = result.get("metadata", {})
            
            print(f"\n📋 Results Summary:")
            print(f"   • Questions processed: {len(answers)}")
            print(f"   • Total processing time: {total_time:.2f}s")
            print(f"   • Average time per question: {total_time/len(answers):.2f}s")
            
            if metadata:
                print(f"   • Processing metadata: {metadata}")
            
            print(f"\n📝 Answers:")
            for i, (question, answer) in enumerate(zip(test_data['questions'], answers), 1):
                print(f"\n{i}. Q: {question}")
                print(f"   A: {answer}")
                
                # Compare with expected answer
                if i <= len(expected_answers):
                    expected = expected_answers[i-1]
                    if len(answer) > 20 and len(expected) > 20:
                        # Simple similarity check
                        common_words = set(answer.lower().split()) & set(expected.lower().split())
                        similarity = len(common_words) / max(len(set(expected.lower().split())), 1)
                        print(f"   📊 Similarity to expected: {similarity:.2f}")
            
            print(f"\n🎯 Performance Summary:")
            print(f"   • Total time: {total_time:.2f}s")
            print(f"   • Questions: {len(answers)}")
            print(f"   • Avg per question: {total_time/len(answers):.2f}s")
            print(f"   • Success rate: {len([a for a in answers if len(a) > 10])}/{len(answers)}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out after 120 seconds")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_batch_api()
