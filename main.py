import logging
import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import config
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingsManager
from src.query_processor import QueryProcessor
from src.retrieval import RetrievalSystem
from src.decision_engine import DecisionEngine
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

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    document_path: Optional[str] = None

class BatchQueryRequest(BaseModel):
    queries: List[str]
    document_path: Optional[str] = None

class SystemStatus(BaseModel):
    is_healthy: bool
    components: dict
    index_stats: dict
    document_count: int

# Global system components
document_loader = None
embeddings_manager = None
query_processor = None
retrieval_system = None
decision_engine = None
response_formatter = None

def initialize_system():
    """Initialize all system components."""
    global document_loader, embeddings_manager, query_processor, retrieval_system, decision_engine, response_formatter
    
    try:
        logger.info("Initializing system components...")
        
        # Validate configuration
        config.validate()
        
        # Initialize components
        document_loader = DocumentLoader()
        embeddings_manager = EmbeddingsManager()
        query_processor = QueryProcessor()
        retrieval_system = RetrievalSystem(embeddings_manager)
        decision_engine = DecisionEngine()
        response_formatter = ResponseFormatter()
        
        # Try to load existing index
        if embeddings_manager.load_index():
            logger.info("Loaded existing FAISS index")
        else:
            logger.info("No existing index found. Please upload documents first.")
        
        logger.info("System initialization completed")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    initialize_system()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LLM Document Processing System",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check system components
        is_healthy = True
        components = {}
        
        # Check embeddings manager
        try:
            stats = embeddings_manager.get_index_stats()
            components['embeddings_manager'] = stats.get('status') != 'no_index'
        except Exception as e:
            components['embeddings_manager'] = False
            is_healthy = False
        
        # Check document loader
        try:
            components['document_loader'] = document_loader is not None
        except Exception as e:
            components['document_loader'] = False
            is_healthy = False
        
        # Check OpenAI API
        try:
            # Simple test call
            test_embedding = embeddings_manager.create_embedding("test")
            components['openai_api'] = len(test_embedding) > 0
        except Exception as e:
            components['openai_api'] = False
            is_healthy = False
        
        # Check FAISS index
        try:
            components['faiss_index'] = embeddings_manager.index is not None
        except Exception as e:
            components['faiss_index'] = False
            is_healthy = False
        
        system_status = {
            'is_healthy': is_healthy,
            'components': components,
            'index_stats': embeddings_manager.get_index_stats(),
            'document_count': len(embeddings_manager.documents) if embeddings_manager.documents else 0
        }
        
        return response_formatter.format_health_response(system_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a single natural language query."""
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Process query
        query_info = query_processor.process_query(request.query)
        
        # Perform contextual retrieval
        retrieved_clauses = retrieval_system.contextual_retrieval(query_info)
        
        # Evaluate coverage decision
        decision_result = decision_engine.evaluate_coverage(query_info, retrieved_clauses)
        
        # Calculate amounts
        amounts = decision_engine.calculate_coverage_amount(decision_result, query_info['entities'])
        decision_result.amounts = amounts
        
        # Format response
        response = response_formatter.format_response(
            request.query,
            query_info,
            decision_result,
            retrieved_clauses,
            request.document_path
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return response_formatter.format_error_response(request.query, str(e))

@app.post("/batch-query")
async def process_batch_queries(request: BatchQueryRequest):
    """Process multiple queries in batch."""
    try:
        logger.info(f"Processing batch queries: {len(request.queries)} queries")
        
        results = []
        for query in request.queries:
            try:
                # Process individual query
                query_info = query_processor.process_query(query)
                retrieved_clauses = retrieval_system.contextual_retrieval(query_info)
                decision_result = decision_engine.evaluate_coverage(query_info, retrieved_clauses)
                
                # Calculate amounts
                amounts = decision_engine.calculate_coverage_amount(decision_result, query_info['entities'])
                decision_result.amounts = amounts
                
                # Format response
                response = response_formatter.format_response(
                    query,
                    query_info,
                    decision_result,
                    retrieved_clauses,
                    request.document_path
                )
                results.append(response)
                
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                error_response = response_formatter.format_error_response(query, str(e))
                results.append(error_response)
        
        return response_formatter.format_batch_response(request.queries, results)
        
    except Exception as e:
        logger.error(f"Error processing batch queries: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a new policy document."""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file
        upload_path = os.path.join(config.DOCUMENTS_PATH, file.filename)
        os.makedirs(config.DOCUMENTS_PATH, exist_ok=True)
        
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Uploaded document: {file.filename}")
        
        # Process document
        documents = document_loader.process_document(upload_path)
        
        # Add to embeddings index
        embeddings_manager.add_documents(documents)
        
        # Save updated index
        embeddings_manager.save_index()
        
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
    """Reindex all documents in the documents directory."""
    try:
        logger.info("Starting document reindexing...")
        
        # Load all documents
        documents = document_loader.load_all_documents()
        
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found to index")
        
        # Build new index
        embeddings_manager.build_index(documents)
        
        # Save index
        embeddings_manager.save_index()
        
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
    """List all processed documents."""
    try:
        documents = embeddings_manager.documents if embeddings_manager.documents else []
        
        # Get unique document names
        doc_names = set()
        for doc in documents:
            doc_name = doc.metadata.get('document_name', 'unknown')
            doc_names.add(doc_name)
        
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
    """Delete a specific document from the index."""
    try:
        if not embeddings_manager.documents:
            raise HTTPException(status_code=404, detail="No documents in index")
        
        # Filter out documents with matching name
        original_count = len(embeddings_manager.documents)
        embeddings_manager.documents = [
            doc for doc in embeddings_manager.documents
            if doc.metadata.get('document_name') != document_name
        ]
        
        if len(embeddings_manager.documents) == original_count:
            raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found")
        
        # Rebuild index with remaining documents
        if embeddings_manager.documents:
            embeddings_manager.build_index(embeddings_manager.documents)
            embeddings_manager.save_index()
        else:
            embeddings_manager.reset_index()
        
        return {
            "message": f"Document '{document_name}' deleted successfully",
            "remaining_documents": len(embeddings_manager.documents)
        }
        
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.get("/stats")
async def get_system_stats():
    """Get system statistics."""
    try:
        return {
            "index_stats": embeddings_manager.get_index_stats(),
            "document_count": len(embeddings_manager.documents) if embeddings_manager.documents else 0,
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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=config.LOG_LEVEL.lower()
    ) 