#!/usr/bin/env python3
"""
Test script for the LLM Document Processing System.
This script tests the core components without requiring documents or API calls.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from config import config
        print("✓ Config module imported successfully")
        
        from document_loader import DocumentLoader
        print("✓ DocumentLoader imported successfully")
        
        from query_processor import QueryProcessor
        print("✓ QueryProcessor imported successfully")
        
        from response_formatter import ResponseFormatter
        print("✓ ResponseFormatter imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from config import config
        
        # Test basic config attributes
        assert hasattr(config, 'OPENAI_API_KEY'), "Missing OPENAI_API_KEY"
        assert hasattr(config, 'OPENAI_MODEL'), "Missing OPENAI_MODEL"
        assert hasattr(config, 'EMBEDDING_MODEL'), "Missing EMBEDDING_MODEL"
        assert hasattr(config, 'CHUNK_SIZE'), "Missing CHUNK_SIZE"
        assert hasattr(config, 'CHUNK_OVERLAP'), "Missing CHUNK_OVERLAP"
        
        print("✓ Configuration attributes present")
        
        # Test config validation (will fail without API key, which is expected)
        try:
            config.validate()
            print("✓ Configuration validation passed")
        except ValueError as e:
            if "OPENAI_API_KEY" in str(e):
                print("⚠ Configuration validation failed (expected without API key)")
            else:
                print(f"✗ Configuration validation failed: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_query_processor():
    """Test query processing functionality."""
    print("\nTesting query processor...")
    
    try:
        from query_processor import QueryProcessor, QueryEntities
        
        # Test without OpenAI API (should use fallback)
        processor = QueryProcessor()
        
        # Test entity extraction
        test_query = "46M, knee surgery, Pune, 3-month policy"
        entities = processor.extract_entities(test_query)
        
        assert isinstance(entities, QueryEntities), "Should return QueryEntities"
        assert entities.age == 46, f"Expected age 46, got {entities.age}"
        assert entities.gender == "male", f"Expected gender male, got {entities.gender}"
        assert entities.procedure == "knee surgery", f"Expected procedure knee surgery, got {entities.procedure}"
        assert entities.location == "Pune", f"Expected location Pune, got {entities.location}"
        
        print("✓ Entity extraction working")
        
        # Test query type detection
        assert entities.query_type == "coverage_check", f"Expected query_type coverage_check, got {entities.query_type}"
        print("✓ Query type detection working")
        
        # Test search query generation
        search_queries = processor.create_search_queries(entities)
        assert isinstance(search_queries, list), "Should return list of search queries"
        assert len(search_queries) > 0, "Should generate at least one search query"
        print("✓ Search query generation working")
        
        return True
    except Exception as e:
        print(f"✗ Query processor test failed: {e}")
        return False

def test_response_formatter():
    """Test response formatting functionality."""
    print("\nTesting response formatter...")
    
    try:
        from response_formatter import ResponseFormatter
        from decision_engine import DecisionResult
        
        formatter = ResponseFormatter()
        
        # Test amount formatting
        test_amounts = {
            'amount': 50000,
            'currency': 'INR'
        }
        
        formatted_amounts = formatter._format_amounts(test_amounts)
        assert formatted_amounts['covered_amount'] == 50000, "Amount formatting failed"
        assert formatted_amounts['currency'] == 'INR', "Currency formatting failed"
        print("✓ Amount formatting working")
        
        # Test metadata formatting
        test_metadata = formatter._format_metadata(2.5, "test.pdf", {})
        assert 'processing_time' in test_metadata, "Metadata formatting failed"
        assert test_metadata['document_source'] == "test.pdf", "Document source formatting failed"
        print("✓ Metadata formatting working")
        
        return True
    except Exception as e:
        print(f"✗ Response formatter test failed: {e}")
        return False

def test_document_loader():
    """Test document loader functionality."""
    print("\nTesting document loader...")
    
    try:
        from document_loader import DocumentLoader
        
        loader = DocumentLoader()
        
        # Test text cleaning
        test_text = "  This   is   a   test   text   with   extra   spaces  "
        cleaned_text = loader._clean_text(test_text)
        assert cleaned_text == "This is a test text with extra spaces", "Text cleaning failed"
        print("✓ Text cleaning working")
        
        # Test section extraction (with mock text)
        mock_text = """
        SECTION A
        This is section A content.
        
        SECTION B
        This is section B content.
        """
        
        sections = loader.extract_sections(mock_text)
        assert len(sections) == 2, f"Expected 2 sections, got {len(sections)}"
        assert sections[0]['section_id'] == 'SECTION A', "Section ID extraction failed"
        assert sections[1]['section_id'] == 'SECTION B', "Section ID extraction failed"
        print("✓ Section extraction working")
        
        return True
    except Exception as e:
        print(f"✗ Document loader test failed: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "data/policies",
        "data/processed",
        "src",
        "tests"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory {dir_path} exists")
        else:
            print(f"✗ Directory {dir_path} missing")
            return False
    
    return True

def test_requirements():
    """Test that required files exist."""
    print("\nTesting required files...")
    
    required_files = [
        "requirements.txt",
        "config.py",
        "main.py",
        "README.md",
        "src/__init__.py",
        "src/document_loader.py",
        "src/embeddings.py",
        "src/query_processor.py",
        "src/retrieval.py",
        "src/decision_engine.py",
        "src/response_formatter.py",
        "tests/test_queries.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ File {file_path} exists")
        else:
            print(f"✗ File {file_path} missing")
            return False
    
    return True

def main():
    """Run all tests."""
    print("LLM Document Processing System - Component Tests")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(level=logging.ERROR)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Required Files", test_requirements),
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Query Processor", test_query_processor),
        ("Response Formatter", test_response_formatter),
        ("Document Loader", test_document_loader),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for setup.")
        print("\nNext steps:")
        print("1. Set your OpenAI API key in .env file")
        print("2. Add PDF documents to data/policies/ directory")
        print("3. Run: python main.py")
        print("4. Visit http://localhost:8000/docs for API documentation")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 