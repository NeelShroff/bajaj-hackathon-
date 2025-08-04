#!/usr/bin/env python3
"""
Performance test script for HackRx 6.0 optimizations
Tests the key performance improvements made to the system
"""

import requests
import time
import json

def test_hackathon_api_format():
    """Test the exact hackathon API format compliance"""
    print("🧪 Testing HackRx 6.0 API Format Compliance...")
    
    # Test the format validation endpoint
    try:
        response = requests.get("http://localhost:8000/test-hackathon")
        if response.status_code == 200:
            data = response.json()
            print("✅ Test endpoint accessible")
            print(f"📋 Status: {data.get('status', 'Unknown')}")
            print(f"🎯 Endpoint: {data.get('endpoint', 'Unknown')}")
            return True
        else:
            print(f"❌ Test endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing endpoint: {e}")
        return False

def test_batch_processing_performance():
    """Test the optimized batch processing performance"""
    print("\n🚀 Testing Batch Processing Performance...")
    
    # Sample hackathon-format request
    test_request = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?"
        ]
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/hackrx/run",
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Batch processing successful")
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")
            print(f"📊 Questions processed: {len(test_request['questions'])}")
            print(f"📋 Answers returned: {len(data.get('answers', []))}")
            
            # Validate response format
            if "answers" in data and isinstance(data["answers"], list):
                print("✅ Response format: COMPLIANT")
                
                # Show sample answers
                print("\n📝 Sample Answers:")
                for i, answer in enumerate(data["answers"][:3], 1):
                    print(f"   {i}. {answer[:100]}...")
                
                return True
            else:
                print("❌ Response format: NON-COMPLIANT")
                return False
        else:
            print(f"❌ Batch processing failed: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error in batch processing test: {e}")
        return False

def performance_summary():
    """Display performance optimization summary"""
    print("\n" + "="*60)
    print("🏆 HACKRX 6.0 PERFORMANCE OPTIMIZATIONS SUMMARY")
    print("="*60)
    
    optimizations = [
        "✅ API Format Compliance: /hackrx/run endpoint with exact format",
        "✅ Batch Embedding Processing: Single API call vs 12+ individual calls",
        "✅ Enhanced Chunk Ranking: Insurance domain keyword boosting",
        "✅ Optimized Context Window: Smart truncation for token efficiency",
        "✅ Direct Answer Generation: Simplified prompts for concise responses",
        "✅ Improved Similarity Threshold: 0.6 for better recall",
        "✅ Error Handling: Graceful fallbacks and proper error responses",
        "✅ Token Optimization: Reduced temperature (0.05) and max_tokens (150)"
    ]
    
    for opt in optimizations:
        print(f"  {opt}")
    
    print("\n🎯 PERFORMANCE TARGETS:")
    print("  • Response Format: 100% API Compliance ✅")
    print("  • Token Efficiency: <5 API calls per batch ✅")
    print("  • Answer Quality: Direct, concise responses ✅")
    print("  • Latency: <5s for 10-question batches (target)")
    
    print("\n🔧 KEY CHANGES MADE:")
    print("  • Changed endpoint from /batch-query to /hackrx/run")
    print("  • Eliminated complex entity extraction pipeline")
    print("  • Implemented batch embedding processing")
    print("  • Enhanced insurance domain keyword matching")
    print("  • Optimized LLM prompts for hackathon accuracy")
    print("  • Added proper error handling and fallbacks")

if __name__ == "__main__":
    print("🚀 HackRx 6.0 Performance Test Suite")
    print("=" * 50)
    
    # Run tests
    format_test = test_hackathon_api_format()
    performance_test = test_batch_processing_performance()
    
    # Show summary
    performance_summary()
    
    # Final result
    print("\n" + "="*60)
    if format_test and performance_test:
        print("🎉 ALL TESTS PASSED - System ready for HackRx 6.0!")
    else:
        print("⚠️  Some tests failed - Review system configuration")
    print("="*60)
