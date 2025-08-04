#!/usr/bin/env python3
"""
LLM-Powered Intelligent Document Query System - FastAPI Server
This server provides an API for document loading, querying, and batch processing.
"""

import logging
import os
import sys
import requests
import tempfile
import asyncio
import uvicorn
import numpy as np
import time # Corrected: Added the missing import for the 'time' module
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

# Ensure the project structure is correct for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import config
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingsManager
from src.retrieval import RetrievalSystem

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Document Processing System",
    description="Intelligent query-retrieval system for insurance policy documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Placeholder Classes (Needed for compilation if not provided) ---
class DecisionEngine:
    def evaluate_coverage(self, query_info, retrieved_clauses):
        return {"result": "Placeholder"}

class ResponseFormatter:
    def format_health_response(self, status):
        return status
    def format_response(self, query, query_info, decision_result, clauses, doc_path):
        return {"answer": "Placeholder", "confidence": 0.5, "sources": [], "reasoning": "Placeholder", "metadata": {}}
    def format_error_response(self, query, error):
        return {"answer": f"Error: {error}"}

class QueryProcessor:
    def process_query(self, query):
        return {"query": query, "entities": {}}
# --- End of Placeholder Classes ---

# Security
security = HTTPBearer(auto_error=False)

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify Bearer token authentication."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    token = credentials.credentials
    if not token or len(token) < 10:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return token

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    document_path: Optional[str] = None

class BatchQueryRequest(BaseModel):
    documents: str = Field(..., description="URL to policy document")
    questions: List[str] = Field(..., description="List of questions to answer")

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="The answer to the question")
    confidence: float = Field(..., description="Confidence score (0-1)")
    sources: List[str] = Field(..., description="Source citations")
    reasoning: Optional[str] = Field(None, description="Reasoning chain")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class BatchQueryResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    processing_time: float = Field(..., description="Processing time in seconds")

class SystemStatus(BaseModel):
    is_healthy: bool
    components: dict
    index_stats: dict
    document_count: int

# Global system components
document_loader: DocumentLoader = DocumentLoader()
embeddings_manager: EmbeddingsManager = EmbeddingsManager()
retrieval_system: RetrievalSystem = RetrievalSystem()
query_processor: QueryProcessor = QueryProcessor()
decision_engine: DecisionEngine = DecisionEngine()
response_formatter: ResponseFormatter = ResponseFormatter()

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    try:
        config.validate()
        logger.info("System initialization started on startup.")
        if embeddings_manager.load_index():
            logger.info("Loaded existing FAISS index on startup.")
        else:
            logger.warning("No existing index found on startup. Please load a document.")
        logger.info("System initialization completed.")
    except Exception as e:
        logger.error(f"System initialization failed: {e}")

@app.get("/")
async def root():
    return {
        "message": "LLM Document Processing System",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    try:
        is_healthy = True
        components = {
            'document_loader': document_loader is not None,
            'embeddings_manager': embeddings_manager is not None,
            'retrieval_system': retrieval_system is not None
        }

        try:
            test_embedding = embeddings_manager.create_embedding("test")
            components['openai_api'] = len(test_embedding) > 0
        except Exception as e:
            components['openai_api'] = False
            is_healthy = False
            logger.error(f"OpenAI API health check failed: {e}")

        try:
            index_stats = embeddings_manager.get_index_stats()
            components['faiss_index'] = index_stats.get('status') != 'no_index'
            document_count = len(embeddings_manager.documents) if embeddings_manager.documents else 0
        except Exception as e:
            components['faiss_index'] = False
            index_stats = {"status": "error"}
            document_count = 0
            is_healthy = False
            logger.error(f"FAISS index health check failed: {e}")

        system_status = {
            'is_healthy': is_healthy,
            'components': components,
            'index_stats': index_stats,
            'document_count': document_count
        }
        
        return response_formatter.format_health_response(system_status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/query")
async def process_query(request: QueryRequest, token: str = Depends(verify_token)):
    try:
        logger.info(f"Processing query: {request.query}")
        
        query_info = query_processor.process_query(request.query)
        
        # Asynchronously retrieve and generate answer
        answer, sources, confidence = await retrieval_system.retrieve_and_generate_answer_async(
            query=request.query, 
            embeddings_manager=embeddings_manager
        )
        
        # This part of the pipeline is currently using placeholder logic
        retrieved_clauses = [{"content": answer, "metadata": {"source": s}} for s in sources]
        decision_result = decision_engine.evaluate_coverage(query_info, retrieved_clauses)
        
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "reasoning": "Answer generated from retrieved context.",
            "metadata": {"query_info": query_info, "decision_result": decision_result}
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/hackrx/run")
async def process_batch_queries(request: BatchQueryRequest, token: str = Depends(verify_token)):
    try:
        start_time = time.time()
        logger.info(f"Processing {len(request.questions)} questions for document: {request.documents}")
        
        if request.documents.startswith('http'):
            try:
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, 'document.pdf')
                
                response = requests.get(request.documents, timeout=30)
                response.raise_for_status()
                
                with open(temp_path, "wb") as f:
                    f.write(response.content)
                
                documents = document_loader.load_document(temp_path)
                if documents:
                    await asyncio.to_thread(embeddings_manager.build_index, documents)
                    logger.info(f"Successfully loaded and indexed document from URL: {len(documents)} chunks")
                else:
                    logger.warning("No content extracted from document URL")
                
                os.remove(temp_path)
                os.rmdir(temp_dir)
            except Exception as e:
                logger.error(f"Error loading document from URL: {e}")
                raise HTTPException(status_code=500, detail=f"Error loading document: {str(e)}")
        
        if embeddings_manager.index is None:
            return {"answers": ["No document index available. Please upload a document first."] * len(request.questions)}
        
        tasks = [
            retrieval_system.retrieve_and_generate_answer_async(q, embeddings_manager)
            for q in request.questions
        ]
        
        results = await asyncio.gather(*tasks)
        
        answers = [res[0] for res in results]
        
        processing_time = time.time() - start_time
        
        return {
            "answers": answers,
            "metadata": {"processing_time": processing_time},
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error processing batch queries: {e}")
        raise HTTPException(status_code=500, detail=f"System error occurred: {str(e)}")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), token: str = Depends(verify_token)):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        temp_dir = tempfile.mkdtemp()
        upload_path = os.path.join(temp_dir, file.filename)
        
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        documents = document_loader.load_document(upload_path)
        await asyncio.to_thread(embeddings_manager.build_index, documents)
        await asyncio.to_thread(embeddings_manager.save_index)
        
        os.remove(upload_path)
        os.rmdir(temp_dir)
        
        return {
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "chunks_created": len(documents),
            "total_documents": len(embeddings_manager.documents)
        }
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/reindex")
async def reindex_documents():
    try:
        logger.info("Starting document reindexing...")
        documents = document_loader.load_all_documents()
        
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found to index")
        
        await asyncio.to_thread(embeddings_manager.build_index, documents)
        await asyncio.to_thread(embeddings_manager.save_index)
        
        return {
            "message": "Documents reindexed successfully",
            "total_documents": len(documents),
            "index_stats": embeddings_manager.get_index_stats()
        }
    except Exception as e:
        logger.error(f"Error reindexing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    try:
        documents = embeddings_manager.documents if embeddings_manager.documents else []
        doc_names = set(doc.metadata.get('document_name', 'unknown') for doc in documents)
        
        return {
            "total_documents": len(documents),
            "unique_documents": len(doc_names),
            "document_names": list(doc_names),
            "index_stats": embeddings_manager.get_index_stats()
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.delete("/documents/{document_name}")
async def delete_document(document_name: str):
    try:
        if not embeddings_manager.documents:
            raise HTTPException(status_code=404, detail="No documents in index")
        
        original_count = len(embeddings_manager.documents)
        embeddings_manager.documents = [
            doc for doc in embeddings_manager.documents
            if doc.metadata.get('document_name') != document_name
        ]
        
        if len(embeddings_manager.documents) == original_count:
            raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found")
        
        if embeddings_manager.documents:
            await asyncio.to_thread(embeddings_manager.build_index, embeddings_manager.documents)
            await asyncio.to_thread(embeddings_manager.save_index)
        else:
            await asyncio.to_thread(embeddings_manager.reset_index)
        
        return {
            "message": f"Document '{document_name}' deleted successfully",
            "remaining_documents": len(embeddings_manager.documents)
        }
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.get("/stats")
async def get_system_stats():
    try:
        index_stats = embeddings_manager.get_index_stats()
        document_count = len(embeddings_manager.documents) if embeddings_manager.documents else 0
        
        return {
            "index_stats": index_stats,
            "document_count": document_count,
            "configuration": {
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "top_k_results": config.TOP_K_RESULTS,
                "similarity_threshold": config.SIMILARITY_THRESHOLD
            }
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.get("/test-hackathon")
async def test_hackathon_format():
    test_request = {
        "documents": "https://example.com/sample-policy.pdf",
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?",
            "Does this policy cover maternity expenses?"
        ]
    }
    test_response = {
        "answers": [
            "A grace period of thirty days is provided for premium payment after the due date.",
            "There is a waiting period of thirty-six (36) months of continuous coverage for pre-existing diseases.",
            "Yes, the policy covers maternity expenses with specific conditions and waiting periods."
        ]
    }
    return {
        "message": "Hackathon API format validation",
        "sample_request": test_request,
        "sample_response": test_response,
        "status": "API format compliant",
        "endpoint": "/hackrx/run",
        "method": "POST"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=config.LOG_LEVEL.lower()
    )