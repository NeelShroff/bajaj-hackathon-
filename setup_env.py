#!/usr/bin/env python3
"""
Environment setup script for HackRx 6.0
Helps configure the required environment variables
"""

import os
from pathlib import Path

def setup_environment():
    """Setup environment variables for the HackRx API"""
    
    env_file = Path(".env")
    
    print("🔧 HackRx 6.0 Environment Setup")
    print("=" * 40)
    
    # Check if .env exists
    if env_file.exists():
        print("✅ .env file found")
        with open(env_file, 'r') as f:
            content = f.read()
            if "API_SECRET_TOKEN" in content:
                print("✅ API_SECRET_TOKEN already configured")
            else:
                print("⚠️  API_SECRET_TOKEN not found in .env")
    else:
        print("⚠️  .env file not found")
    
    # Check current environment
    api_token = os.getenv("API_SECRET_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    
    print("\n📋 Current Environment Status:")
    print(f"API_SECRET_TOKEN: {'✅ Set' if api_token and api_token != 'YOUR_SECURE_TOKEN_HERE' else '❌ Not set'}")
    print(f"OPENAI_API_KEY: {'✅ Set' if openai_key and openai_key != 'YOUR_OPENAI_API_KEY_HERE' else '❌ Not set'}")
    print(f"PINECONE_API_KEY: {'✅ Set' if pinecone_key and pinecone_key != 'YOUR_PINECONE_API_KEY_HERE' else '❌ Not set'}")
    
    # For testing purposes, set a default API token if not set
    if not api_token or api_token == "YOUR_SECURE_TOKEN_HERE":
        print("\n🔑 Setting default API token for testing...")
        os.environ["API_SECRET_TOKEN"] = "hackrx-test-token-2024"
        print("✅ API_SECRET_TOKEN set to: hackrx-test-token-2024")
        
        # Update .env file
        env_content = ""
        if env_file.exists():
            with open(env_file, 'r') as f:
                env_content = f.read()
        
        if "API_SECRET_TOKEN" not in env_content:
            with open(env_file, 'a') as f:
                f.write("\n# API Authentication Token\n")
                f.write("API_SECRET_TOKEN=hackrx-test-token-2024\n")
            print("✅ Added API_SECRET_TOKEN to .env file")
    
    print("\n🚀 Environment setup complete!")
    print("📝 Use token 'hackrx-test-token-2024' for testing")
    
    return os.getenv("API_SECRET_TOKEN")

if __name__ == "__main__":
    token = setup_environment()
    print(f"\n🔑 Current API Token: {token}")
