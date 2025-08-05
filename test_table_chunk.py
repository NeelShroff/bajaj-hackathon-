#!/usr/bin/env python3
"""
API test script to debug the retrieval process for a specific query.
"""

import requests
import json
import time

# IMPORTANT: Update this token to match the secret token in your main.py server
API_SECRET_TOKEN = "YOUR_SECURE_TOKEN_HERE"

def debug_query_retrieval(query: str):
    """
    Sends a query to the /debug-retrieval endpoint and prints the results.
    """
    print("\n\n--- Debugging Retrieval for a Specific Query ---")
    print(f"Query: '{query}'")
    print("=" * 60)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_SECRET_TOKEN}"
    }

    request_data = {
        "query": query
    }
    
    try:
        # Send a POST request to your FastAPI server's debugging endpoint
        response = requests.post(
            "http://localhost:8000/debug-retrieval", 
            json=request_data, 
            headers=headers, 
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"✅ Response received with status: {response.status_code}")
            chunks = response.json()
            print(f"Retrieved {len(chunks)} chunks.")
            print("-" * 60)
            
            # Print each chunk for inspection
            for i, chunk in enumerate(chunks):
                metadata = chunk['metadata']
                print(f"Chunk {i+1} (Confidence: {metadata.get('confidence'):.2f}, Type: {metadata.get('section_type')})")
                print(f"--- Content (starts) ---")
                print(chunk['content'])
                print(f"--- Content (ends) ---")
                
                if 'row_data' in metadata:
                    print("\n--- Raw Table Row Data (for verification) ---")
                    print(json.dumps(metadata['row_data'], indent=2))
                
                print("-" * 60)
            
        else:
            print(f"❌ Error: Received status code {response.status_code}")
            print(f"Response body: {response.text}")
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("Please ensure the FastAPI server is running at http://localhost:8000")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    # The specific question you want to test
    test_query = "What is the coverage limit for air ambulance services under Plan B?"
    debug_query_retrieval(test_query)