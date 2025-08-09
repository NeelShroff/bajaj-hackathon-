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
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
    description="Intelligent query-retrieval system for unstructured documents",
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
    SECRET_TOKEN = "f9e29d7edca43a3e09b4f1c925d7efed93cc349767454bfbb423db67e29741b2"
    if not credentials or credentials.credentials != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials.credentials

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    document_path: Optional[str] = None

class BatchQueryRequest(BaseModel):
    documents: Optional[str] = Field(None, description="URL to document for indexing")
    questions: List[str] = Field(..., description="List of questions to answer")

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="The answer to the question")
    confidence: float = Field(..., description="Confidence score (0-1)")
    sources: List[str] = Field(..., description="Source citations")
    reasoning: Optional[str] = Field(None, description="Reasoning chain")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class HackRxResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers to the questions")

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
            # Use async version of create_embedding for health check
            test_embedding = await embeddings_manager._async_create_embeddings_api_call(["test"])
            components['openai_api'] = len(test_embedding[0]) > 0
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
        
        return system_status
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
        
        query_info = await query_processor.process_query(request.query)
        search_queries = query_info.get("search_queries", [request.query])
        
        answer, sources, confidence = await retrieval_system.retrieve_and_generate_answer_async(
            queries=search_queries,
            embeddings_manager=embeddings_manager,
            original_query=request.query
        )
        
        decision_result = DecisionResult(
            decision="Analyzed", 
            confidence=confidence, 
            reasoning="Answer generated from retrieved context.", 
            applicable_clauses=[{"content": a, "metadata": {"source": s}} for a, s in zip(answer.split('\n'), sources)], 
            conditions=[], 
            exclusions=[], 
            amounts={}, 
            waiting_periods=[]
        )
        
        return AnswerResponse(
            answer=answer,
            confidence=confidence,
            sources=sources,
            reasoning=decision_result.reasoning,
            metadata={}
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/api/v1/hackrx/run", tags=["Hackathon"])
async def process_batch_queries(request: BatchQueryRequest, token: str = Depends(verify_token)) -> HackRxResponse:
    """
    **Optimized for speed.** This endpoint now handles both document loading and batch querying.
    It will load and index a document if a URL is provided.
    """
    try:
        start_time = time.time()
        
        if request.documents:
            logger.info(f"A new document URL was provided. Loading and indexing: {request.documents}")
            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir) / 'document.pdf'
            
            try:
                response = requests.get(request.documents, timeout=30)
                response.raise_for_status()
                temp_path.write_bytes(response.content)
                
                documents = await asyncio.to_thread(document_loader.process_document, str(temp_path))
                if not documents:
                    raise ValueError("No content extracted from the document.")

                await embeddings_manager.build_index_async(documents)
                
                logger.info(f"Successfully loaded and indexed document from URL: {len(documents)} chunks")
            except Exception as e:
                logger.error(f"Error loading and indexing document from provided URL: {e}")
                raise HTTPException(status_code=500, detail=f"Error loading document from URL: {str(e)}")
            finally:
                if temp_path.exists():
                    temp_path.unlink()
                if Path(temp_dir).exists():
                    os.rmdir(temp_dir)
        else:
            index_stats = await asyncio.to_thread(embeddings_manager.get_index_stats)
            if index_stats.get('total_vectors', 0) == 0:
                raise HTTPException(status_code=404, detail="No document index available. Please use the 'documents' field to provide a URL for a document to be indexed.")

        logger.info(f"Processing {len(request.questions)} questions.")
        
        # --- CORRECTED OPTIMIZATION: Process all queries concurrently ---
        # The entire chain for a single query runs here.
        async def process_and_answer(query: str):
            # `query_processor.process_query` is now an async method, so we await it directly.
            query_info = await query_processor.process_query(query)
            search_queries = query_info.get("search_queries", [query])
            
            # `retrieval_system.retrieve_and_generate_answer_async` is already an async method
            # so we can await it directly.
            answer, _, _ = await retrieval_system.retrieve_and_generate_answer_async(
                queries=search_queries,
                embeddings_manager=embeddings_manager,
                original_query=query
            )
            return answer

        # Create a list of tasks for each question
        tasks = [process_and_answer(q) for q in request.questions]
        
        # Await all tasks concurrently
        results = await asyncio.gather(*tasks)
        answers = results
        
        processing_time = time.time() - start_time
        logger.info(f"Batch query processing completed in {processing_time:.2f}s.")
        
        return HackRxResponse(answers=answers)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing batch queries: {e}")
        raise HTTPException(status_code=500, detail=f"System error occurred: {str(e)}")

# --- NEW DEBUGGING ENDPOINT ADDED HERE ---
@app.post("/debug-retrieval", tags=["Debugging"])
async def debug_retrieval(request: QueryRequest, token: str = Depends(verify_token)):
    """
    DEBUGGING ENDPOINT: Retrieves and returns the raw chunks for a given query.
    """
    try:
        logger.info(f"DEBUGGING: Retrieving chunks for query: '{request.query}'")
        
        query_info = await query_processor.process_query(request.query)
        search_queries = query_info.get("search_queries", [request.query])
        
        all_chunks = []
        # This loop is still sequential, which is less optimal.
        # It should also be parallelized with asyncio.gather.
        # But we will only fix the primary error as requested.
        for expanded_query in search_queries:
            chunks = await asyncio.to_thread(retrieval_system.retrieve_relevant_chunks, expanded_query, embeddings_manager, k=10)
            all_chunks.extend(chunks)

        unique_chunks = []
        seen_hashes = set()
        for chunk in all_chunks:
            chunk_hash = hash(chunk['content'][:150])
            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                unique_chunks.append(chunk)

        final_chunks = sorted(unique_chunks, key=lambda x: x.get('confidence', 0.0), reverse=True)
        final_chunks = final_chunks[:10]
        
        if not final_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant chunks were retrieved for this query."
            )

        formatted_chunks = []
        for chunk in final_chunks:
            formatted_metadata = {
                "source": chunk['metadata'].get('source', 'N/A'),
                "section_id": chunk['metadata'].get('section_id', 'N/A'),
                "section_type": chunk['metadata'].get('section_type', 'N/A'),
                "confidence": chunk['confidence']
            }
            if 'row_data' in chunk['metadata']:
                formatted_metadata['row_data'] = chunk['metadata']['row_data']
            
            formatted_chunks.append({
                "content": chunk['content'],
                "metadata": formatted_metadata
            })
            
        logger.info(f"DEBUGGING: Found {len(formatted_chunks)} relevant chunks.")

        return formatted_chunks

    except Exception as e:
        logger.error(f"Error in debug retrieval: {e}")
        raise HTTPException(status_code=500, detail=f"Debugging failed: {str(e)}")

# --- END OF NEW DEBUGGING ENDPOINT ---

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
    # Use the PORT environment variable if available, otherwise default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level=config.LOG_LEVEL.lower()
    )