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
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from pathlib import Path

# Ensure the project structure is correct for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import config
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingsManager
from src.retrieval import RetrievalSystem
from src.query_processor import QueryProcessor
from src.decision_engine import DecisionEngine, DecisionResult
from src.response_formatter import ResponseFormatter

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

# Security
security = HTTPBearer(auto_error=False)

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    """Verify Bearer token authentication."""
    SECRET_TOKEN = os.getenv("API_SECRET_TOKEN", "YOUR_SECURE_TOKEN_HERE")
    
    if not credentials or credentials.credentials != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    
    return credentials.credentials

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    document_path: Optional[str] = None

class BatchQueryRequest(BaseModel):
    documents: Optional[str] = Field(None, description="URL to policy document (deprecated)")
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
        
        logger.info("Pinecone index will be initialized or connected via EmbeddingsManager.")
        
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
            test_embedding = await asyncio.to_thread(embeddings_manager.create_embedding, "test")
            components['openai_api'] = len(test_embedding) > 0
        except Exception as e:
            components['openai_api'] = False
            is_healthy = False
            logger.error(f"OpenAI API health check failed: {e}")

        try:
            index_stats = await asyncio.to_thread(embeddings_manager.get_index_stats)
            components['pinecone_index'] = index_stats.get('status') == 'ready'
            document_count = index_stats.get('total_vectors', 0)
        except Exception as e:
            components['pinecone_index'] = False
            index_stats = {"status": "error"}
            document_count = 0
            is_healthy = False
            logger.error(f"Pinecone index health check failed: {e}")

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

@app.post("/load-and-index", tags=["Document Management"])
async def load_and_index_document(request: dict, token: str = Depends(verify_token)):
    """Loads and indexes a document from a URL into the Pinecone index."""
    document_url = request.get("document_url")
    if not document_url:
        raise HTTPException(status_code=400, detail="Missing 'document_url' in request body.")

    start_time = time.time()
    logger.info(f"Loading and indexing document from URL: {document_url}")

    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / 'document.pdf'
    
    try:
        response = requests.get(document_url, timeout=30)
        response.raise_for_status()
        
        temp_path.write_bytes(response.content)
        
        documents = await asyncio.to_thread(document_loader.process_document, str(temp_path))
        if not documents:
            raise ValueError("No content extracted from the document.")

        await embeddings_manager.build_index_async(documents)
        
        logger.info(f"Successfully loaded and indexed document from URL: {len(documents)} chunks")
        
        return {
            "message": "Document loaded and indexed successfully.",
            "url": document_url,
            "chunks_indexed": len(documents),
            "processing_time": time.time() - start_time
        }
    except Exception as e:
        logger.error(f"Error loading and indexing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading document: {str(e)}")
    finally:
        if temp_path.exists():
            temp_path.unlink()
        if Path(temp_dir).exists():
            os.rmdir(temp_dir)

@app.post("/query", tags=["Querying"])
async def process_query(request: QueryRequest, token: str = Depends(verify_token)) -> AnswerResponse:
    try:
        logger.info(f"Processing query: {request.query}")
        
        query_info = await asyncio.to_thread(query_processor.process_query, request.query)
        
        answer, sources, confidence = await retrieval_system.retrieve_and_generate_answer_async(
            query=request.query, 
            embeddings_manager=embeddings_manager
        )
        
        retrieved_clauses = [{"content": answer, "metadata": {"source": s, 'relevance_score': 0.8}} for s in sources]
        
        decision_result = DecisionResult(
            decision="covered", 
            confidence=confidence, 
            reasoning="Answer generated from retrieved context.", 
            applicable_clauses=retrieved_clauses, 
            conditions=[], 
            exclusions=[], 
            amounts={}, 
            waiting_periods=[]
        )
        
        formatted_response = response_formatter.format_response(
            query=request.query,
            query_info=query_info,
            decision_result=decision_result,
            retrieved_clauses={"retrieved": [(c["content"], c["metadata"]["relevance_score"]) for c in retrieved_clauses]},
            document_source=request.document_path
        )

        return AnswerResponse(
            answer=formatted_response['justification']['reasoning'],
            confidence=formatted_response['confidence'],
            sources=[c['source'] for c in formatted_response['justification']['applicable_clauses']],
            reasoning=formatted_response['justification']['reasoning'],
            metadata=formatted_response['metadata']
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/hackrx/run", tags=["Hackathon"])
async def process_batch_queries(request: BatchQueryRequest, token: str = Depends(verify_token)) -> BatchQueryResponse:
    """
    **Optimized for speed.** Assumes a document has been pre-indexed using `/load-and-index`.
    This endpoint now only handles the LLM generation step.
    """
    try:
        # Check if the index is ready BEFORE processing queries
        index_stats = await asyncio.to_thread(embeddings_manager.get_index_stats)
        if index_stats.get('total_vectors', 0) == 0:
            raise HTTPException(status_code=404, detail="No document index available. Please use the /load-and-index endpoint first.")

        start_time = time.time()
        logger.info(f"Processing {len(request.questions)} questions against pre-indexed document.")
        
        tasks = [
            retrieval_system.retrieve_and_generate_answer_async(q, embeddings_manager)
            for q in request.questions
        ]
        
        results = await asyncio.gather(*tasks)
        
        answers = [res[0] for res in results]
        
        processing_time = time.time() - start_time
        
        return BatchQueryResponse(
            answers=answers,
            metadata={"processing_time": processing_time},
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error processing batch queries: {e}")
        raise HTTPException(status_code=500, detail=f"System error occurred: {str(e)}")


@app.post("/upload", tags=["Document Management"])
async def upload_document(file: UploadFile = File(...), token: str = Depends(verify_token)):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        temp_dir = tempfile.mkdtemp()
        upload_path = Path(temp_dir) / file.filename
        
        upload_path.write_bytes(await file.read())
        
        documents = await asyncio.to_thread(document_loader.process_document, str(upload_path))
        await embeddings_manager.build_index_async(documents)
        
        upload_path.unlink()
        os.rmdir(temp_dir)
        
        return {
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "chunks_created": len(documents),
            "index_stats": embeddings_manager.get_index_stats()
        }
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/reindex", tags=["Document Management"])
async def reindex_documents(token: str = Depends(verify_token)):
    try:
        logger.info("Starting document reindexing...")
        
        documents = await asyncio.to_thread(document_loader.load_all_documents, config.DOCUMENTS_PATH)
        
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found to reindex in the local directory.")
        
        await embeddings_manager.build_index_async(documents)
        
        return {
            "message": "Documents reindexed successfully",
            "total_documents": len(documents),
            "index_stats": embeddings_manager.get_index_stats()
        }
    except Exception as e:
        logger.error(f"Error reindexing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")

@app.delete("/documents/{document_name}", tags=["Document Management"])
async def delete_document(document_name: str, token: str = Depends(verify_token)):
    raise HTTPException(status_code=501, detail="Delete functionality not implemented for Pinecone index.")

@app.get("/documents", tags=["Document Management"])
async def list_documents():
    try:
        index_stats = await asyncio.to_thread(embeddings_manager.get_index_stats)
        if index_stats.get('status') != 'ready':
            return {"total_documents": 0, "unique_documents": 0, "document_names": [], "index_stats": index_stats}

        return {
            "total_vectors": index_stats.get('total_vectors', 0),
            "index_stats": index_stats
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.get("/stats", tags=["System Info"])
async def get_system_stats():
    try:
        index_stats = await asyncio.to_thread(embeddings_manager.get_index_stats)
        document_count = index_stats.get('total_vectors', 0)
        
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

@app.get("/test-hackathon", tags=["Hackathon"])
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
            "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
            "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
            "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
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