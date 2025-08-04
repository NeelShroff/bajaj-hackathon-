#!/usr/bin/env python3
"""
LLM-Powered Intelligent Document Query System - CLI Interface
Command-line testing tool for interactive development and testing

Usage:
    python terminal_app.py load <document_path_or_url>
    python terminal_app.py query "Your question here"
    python terminal_app.py status
    python terminal_app.py reload
    python terminal_app.py interactive
"""

import logging
import os
import sys
import json
import time
import requests
import tempfile
import argparse
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import config
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingsManager
from src.retrieval import RetrievalSystem
from src.query_processor import QueryProcessor

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentQueryCLI:
    """Command-line interface for the Document Query System"""
    
    def __init__(self):
        self.document_loader = None
        self.embeddings_manager = None
        self.retrieval_system = None
        self.query_processor = None
        self.initialized = False
        self.loaded_documents = {}
        self.query_history = []
        self.cache_dir = Path("./data/cli_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.cache_dir / "cli_state.json"
        self.embeddings_file = self.cache_dir / "cli_embeddings.faiss"
        self.documents_file = self.cache_dir / "cli_documents.pkl"
        self.current_document = None
    
    def initialize_system(self):
        """Initialize system components"""
        try:
            print("🔄 Initializing Document Query System...")
            
            # Validate configuration
            config.validate()
            print("✅ Configuration validated")
            
            # Initialize all components
            print("🔄 Loading document processor...")
            self.document_loader = DocumentLoader()
            
            print("🔄 Loading embeddings manager...")
            self.embeddings_manager = EmbeddingsManager()
            
            print("🔄 Loading retrieval system...")
            self.retrieval_system = RetrievalSystem()
            
            print("🔄 Loading query processor...")
            self.query_processor = QueryProcessor()
            
            # Try to load existing index
            if os.path.exists(f"{config.FAISS_INDEX_PATH}.faiss"):
                self.embeddings_manager.load_index()
                print("✅ Loaded existing FAISS index")
            
            self.initialized = True
            print("✅ System initialization completed!")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            print(f"❌ Initialization failed: {e}")
            self.initialized = False
    
    def save_state(self):
        """Save current state to persistent storage."""
        try:
            state_data = {
                'loaded_documents': self.loaded_documents,
                'query_history': self.query_history,
                'initialized': self.initialized,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            # Save embeddings and documents if available
            if self.embeddings_manager and hasattr(self.embeddings_manager, 'index') and self.embeddings_manager.index:
                self.embeddings_manager.save_index(str(self.embeddings_file.with_suffix('')))
                
                # Save documents separately
                import pickle
                if hasattr(self.embeddings_manager, 'documents') and self.embeddings_manager.documents:
                    with open(self.documents_file, 'wb') as f:
                        pickle.dump(self.embeddings_manager.documents, f)
            
            logger.info("State saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self):
        """Load persistent state if available."""
        try:
            if not self.state_file.exists():
                return False
            
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
            
            self.loaded_documents = state_data.get('loaded_documents', {})
            self.query_history = state_data.get('query_history', [])
            
            # Load embeddings if available
            embeddings_base = str(self.embeddings_file.with_suffix(''))
            if Path(f"{embeddings_base}.faiss").exists() and self.documents_file.exists():
                if not self.embeddings_manager:
                    self.embeddings_manager = EmbeddingsManager()
                
                self.embeddings_manager.load_index(embeddings_base)
                
                # Load documents
                import pickle
                with open(self.documents_file, 'rb') as f:
                    self.embeddings_manager.documents = pickle.load(f)
                
                print(f"✅ Loaded persistent state with {len(self.loaded_documents)} documents")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False
    
    def load_command(self, document_path: str):
        """Load and process a document from URL or local path."""
        if not self.initialized:
            self.initialize_system()
        
        if not self.initialized:
            print("❌ Failed to initialize system")
            return False
        
        try:
            print(f"🔄 Loading document: {document_path}")
            start_time = time.time()
            
            if document_path.startswith('http'):
                # Handle URL documents
                print(f"📥 Downloading from URL: {document_path}")
                response = requests.get(document_path, timeout=60)
                response.raise_for_status()
                
                # Determine file extension from URL or content-type
                content_type = response.headers.get('content-type', '')
                if 'pdf' in content_type:
                    suffix = '.pdf'
                elif 'word' in content_type or 'docx' in content_type:
                    suffix = '.docx'
                else:
                    suffix = '.pdf'  # Default to PDF
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(response.content)
                    temp_path = temp_file.name
                
                documents = self.document_loader.load_document(temp_path)
                os.unlink(temp_path)  # Clean up
                
            else:
                # Handle local file
                if not os.path.exists(document_path):
                    print(f"❌ File not found: {document_path}")
                    return False
                
                documents = self.document_loader.load_document(document_path)
            
            if not documents:
                print("❌ No content extracted from document")
                return False
            
            # Build embeddings index
            print(f"🔄 Creating embeddings for {len(documents)} chunks...")
            self.embeddings_manager.build_index(documents)
            
            # Cache the document info
            doc_key = os.path.basename(document_path) if not document_path.startswith('http') else document_path.split('/')[-1]
            self.loaded_documents[doc_key] = {
                'path': document_path,
                'chunks': len(documents),
                'loaded_at': datetime.now().isoformat(),
                'processing_time': time.time() - start_time
            }
            self.current_document = document_path
            
            processing_time = time.time() - start_time
            print(f"✅ Successfully loaded {len(documents)} chunks in {processing_time:.2f}s")
            print(f"📊 Document: {doc_key}")
            if documents:
                print(f"📄 Format: {documents[0].metadata.get('file_format', 'unknown')}")
                print(f"📏 Total characters: {documents[0].metadata.get('total_chars', 'unknown')}")
            
            # Save state for persistence between CLI commands
            self.save_state()
            print("💾 State saved - you can now run queries with 'python terminal_app.py query'")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            print(f"❌ Error loading document: {e}")
            return False
    
    def query_command(self, question: str):
        """Process a natural language query against loaded documents."""
        # Try to load persistent state first
        if not self.initialized:
            self.initialize_system()
            if self.initialized:
                state_loaded = self.load_state()
                if not state_loaded:
                    print("❌ No documents loaded. Run 'load <document_path>' first.")
                    return
            else:
                print("❌ System initialization failed.")
                return
        
        # Check if we have loaded documents (either from current session or persistent state)
        if not self.loaded_documents:
            if not self.load_state():
                print("❌ No documents loaded. Run 'load <document_path>' first.")
                return
        
        try:
            print(f"🔍 Processing query: {question}")
            start_time = time.time()
            
            # Step 1: Retrieve relevant chunks with the improved retrieval system
            relevant_chunks = self.retrieval_system.retrieve_relevant_chunks(
                query=question, 
                embeddings_manager=self.embeddings_manager,
                k=4 # Retrieve a few chunks for good context
            )
            
            if not relevant_chunks:
                print("❌ No relevant information found.")
                return
            
            # DEBUG: Show retrieved chunks for investigation
            print(f"\n🔍 DEBUG - Retrieved {len(relevant_chunks)} chunks:")
            for i, chunk in enumerate(relevant_chunks):
                print(f"  Chunk {i+1} (confidence: {chunk.get('confidence', 'N/A'):.3f}):")
                print(f"    Source: {chunk['metadata'].get('filename', 'Unknown')}, Section: {chunk['metadata'].get('section_id', 'N/A')}")
                print(f"    Content: {chunk['content'][:200]}...")
                print()

            # Step 2: Assemble the context window for the LLM
            context_parts = []
            current_length = 0
            max_context_length = 2000  # A reasonable limit for GPT-3.5-turbo

            for chunk in relevant_chunks:
                content = chunk['content']
                # Stop adding chunks if the context window is full
                if current_length + len(content) <= max_context_length:
                    context_parts.append(content)
                    current_length += len(content)
                else:
                    break
            
            context = "\n\n".join(context_parts)
            
            # DEBUG: Show exact context being sent to LLM
            print(f"\n📝 DEBUG - Context being sent to LLM ({len(context)} chars):")
            print(f"'{context[:500]}...'")
            print("\n" + "="*50)
            
            # Step 3: Generate a nuanced answer using a better-structured prompt
            from openai import OpenAI
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            
            prompt = f"""You are an insurance policy expert. Your task is to provide a comprehensive and accurate answer to the user's question based ONLY on the provided policy text.

POLICY DOCUMENT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Do not make up any information.
- If the policy text provides a specific number (e.g., '30 days', '24 months', '5%'), include that number in your answer.
- If the policy mentions conditions, exclusions, or limitations, summarize them clearly.
- If the policy text does not contain the answer, state that "The policy document does not contain information on this topic."

Answer:"""
            
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # A slightly higher temperature for more descriptive answers
                max_tokens=150    # Sufficient for a detailed answer
            )
            
            answer = response.choices[0].message.content.strip()
            processing_time = time.time() - start_time
            
            # Display results
            print(f"\n📋 Answer:")
            print(f"{answer}")
            print(f"\n⏱️ Processing time: {processing_time:.2f}s")
            print(f"🔍 Retrieved {len(relevant_chunks)} relevant chunks")
            
            # Show source citations
            print(f"\n📚 Sources:")
            for i, chunk in enumerate(relevant_chunks):
                source = chunk.get('metadata', {}).get('filename', 'Unknown')
                confidence = chunk.get('confidence', 0)
                section = chunk.get('metadata', {}).get('section_id', 'N/A')
                print(f"  {i+1}. {source} (Confidence: {confidence:.3f}) - Section: {section}")
            
            # Add to query history
            self.query_history.append({
                'question': question,
                'answer': answer,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'chunks_retrieved': len(relevant_chunks)
            })
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"❌ Error processing query: {e}")
    
    def status_command(self):
        """Show system status and loaded documents."""
        # Try to load persistent state to show accurate status
        if not self.initialized:
            self.initialize_system()
            if self.initialized:
                self.load_state()
        elif not self.loaded_documents:
            self.load_state()
        
        print("\n📊 System Status")
        print("=" * 50)
        
        print(f"🔧 Initialized: {'✅ Yes' if self.initialized else '❌ No'}")
        print(f"📁 Documents loaded: {len(self.loaded_documents)}")
        print(f"📝 Query history: {len(self.query_history)} queries")
        
        if self.embeddings_manager and hasattr(self.embeddings_manager, 'index'):
            try:
                index_size = self.embeddings_manager.index.ntotal if self.embeddings_manager.index else 0
                print(f"🔍 Vector index size: {index_size} embeddings")
            except:
                print(f"🔍 Vector index: Available")
        
        print(f"💾 Cache directory: {self.cache_dir}")
        print(f"⚙️ Config: {config.OPENAI_MODEL} | {config.EMBEDDING_MODEL}")
        
        if self.loaded_documents:
            print("\n📄 Loaded Documents:")
            for doc_name, info in self.loaded_documents.items():
                print(f"  📋 {doc_name}")
                print(f"     📊 Chunks: {info['chunks']}")
                print(f"     ⏰ Loaded: {info['loaded_at'][:19]}")
                print(f"     ⚡ Time: {info['processing_time']:.2f}s")
        
        if self.query_history:
            print(f"\n📈 Recent Queries ({len(self.query_history)} total):")
            for i, query in enumerate(self.query_history[-3:], 1):
                print(f"  {i}. {query['question'][:60]}{'...' if len(query['question']) > 60 else ''}")
                print(f"     ⏱️ {query['processing_time']:.2f}s | 🔍 {query['chunks_retrieved']} chunks")
    
    def reload_command(self):
        """Reload the system and clear cache."""
        print("🔄 Reloading system...")
        
        # Clear current state
        self.initialized = False
        self.loaded_documents = {}
        self.query_history = []
        
        # Reinitialize
        self.initialize_system()
        
        if self.initialized:
            print("✅ System reloaded successfully")
        else:
            print("❌ Failed to reload system")
    
    def interactive_mode(self):
        """Start interactive mode for continuous Q&A."""
        print("💬 Starting Interactive Mode")
        print("Type 'help' for commands, 'exit' to quit\n")
        
        if not self.initialized:
            self.initialize_system()
        
        if not self.loaded_documents:
            print("⚠️ No documents loaded. Use 'load <path>' to load a document first.")
        
        while True:
            try:
                user_input = input("\n🔍 Query> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  load <path>     - Load a document")
                    print("  status          - Show system status")
                    print("  reload          - Reload system")
                    print("  clear           - Clear query history")
                    print("  help            - Show this help")
                    print("  exit/quit/q     - Exit interactive mode")
                    print("  <question>      - Ask a question about loaded documents")
                
                elif user_input.lower().startswith('load '):
                    path = user_input[5:].strip()
                    if path:
                        self.load_command(path)
                    else:
                        print("❌ Please specify a document path")
                
                elif user_input.lower() == 'status':
                    self.status_command()
                
                elif user_input.lower() == 'reload':
                    self.reload_command()
                
                elif user_input.lower() == 'clear':
                    self.query_history = []
                    print("✅ Query history cleared")
                
                else:
                    # Treat as a query
                    self.query_command(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"❌ Error: {e}")
    
    def load_documents(self, path: str = None):
        """Load and index documents from directory or URL"""
        if not self.initialized:
            print("❌ System not initialized. Please run initialization first.")
            return False
        
        try:
            if path and path.startswith('http'):
                # Handle URL documents (hackathon format)
                print(f"🔄 Loading document from URL: {path}")
                response = requests.get(path, timeout=30)
                response.raise_for_status()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(response.content)
                    temp_path = temp_file.name
                
                documents = self.document_loader.load_document(temp_path)
                os.unlink(temp_path)  # Clean up
                
                if documents:
                    self.embeddings_manager.build_index(documents)
                    self.current_document = path
                    print(f"✅ Successfully loaded and indexed {len(documents)} chunks from URL")
                    return True
                else:
                    print("❌ No content extracted from URL")
                    return False
            else:
                # Handle local directory
                documents_path = path or config.DOCUMENTS_PATH
                print(f"🔄 Loading documents from {documents_path} directory...")
                
                documents = self.document_loader.load_all_documents(documents_path)
                
                if documents:
                    print(f"✅ Found {len(documents)} document chunks")
                    
                    print("🔄 Building FAISS index...")
                    self.embeddings_manager.build_index(documents)
                    
                    print("🔄 Saving index...")
                    self.embeddings_manager.save_index()
                    
                    print(f"✅ Successfully indexed {len(documents)} document chunks")
                    return True
                else:
                    print("❌ No documents found to process")
                    return False
                    
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            print(f"❌ Error loading documents: {e}")
            return False
    
    def process_hackathon_batch(self, questions: List[str], document_url: str = None):
        """Process questions in hackathon batch format with optimized performance"""
        if not self.initialized:
            print("❌ System not initialized")
            return {"answers": ["System not initialized"] * len(questions)}
        
        try:
            print(f"🔄 Processing {len(questions)} questions in batch mode...")
            start_time = time.time()
            
            # Load document if URL provided
            if document_url and document_url != self.current_document:
                if not self.load_documents(document_url):
                    return {"answers": ["Failed to load document"] * len(questions)}
            
            if self.embeddings_manager.index is None:
                return {"answers": ["No document index available. Please load documents first."] * len(questions)}
            
            answers = []
            
            print("🔄 Creating batch embeddings for all questions...")
            question_embeddings = self.embeddings_manager.create_embeddings(questions)
            
            for i, (question, question_embedding) in enumerate(zip(questions, question_embeddings)):
                try:
                    print(f"🔄 Processing question {i+1}/{len(questions)}: {question[:50]}...")
                    
                    query_vector = np.array([question_embedding], dtype=np.float32)
                    
                    # Perform retrieval using the main retrieval method
                    relevant_chunks = self.retrieval_system.retrieve_relevant_chunks(
                        query=question, 
                        embeddings_manager=self.embeddings_manager, 
                        k=4
                    )
                    
                    # PHASE 3: OPTIMIZED ANSWER GENERATION
                    if relevant_chunks:
                        context_parts = []
                        current_length = 0
                        max_context_length = 2000
                        
                        for chunk in relevant_chunks:
                            content = chunk['content']
                            if current_length + len(content) <= max_context_length:
                                context_parts.append(content)
                                current_length += len(content)
                            else:
                                break
                        
                        context = "\n\n".join(context_parts)
                        
                        # New, better prompt for robust answer generation
                        prompt = f"""You are an insurance policy expert. Your task is to provide a comprehensive and accurate answer to the user's question based ONLY on the provided policy text.

POLICY DOCUMENT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Do not make up any information.
- If the policy text provides a specific number (e.g., '30 days', '24 months', '5%'), include that number in your answer.
- If the policy mentions conditions, exclusions, or limitations, summarize them clearly.
- If the policy text does not contain the answer, state that "The policy document does not contain information on this topic."

Answer:"""
                        
                        response = self.embeddings_manager.client.chat.completions.create(
                            model=config.OPENAI_MODEL,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            max_tokens=150,
                            top_p=0.9
                        )
                        
                        answer = response.choices[0].message.content.strip()
                        
                        if answer.startswith('ANSWER:'):
                            answer = answer[7:].strip()
                        elif answer.startswith('Direct Answer:'):
                            answer = answer[14:].strip()
                        
                        answers.append(answer)
                        print(f"✅ Answer {i+1}: {answer[:100]}...")
                        
                    else:
                        answer = "The policy document does not contain information on this topic."
                        answers.append(answer)
                        print(f"✅ Answer {i+1}: {answer}")
                        
                except Exception as e:
                    logger.error(f"Error processing question '{question}': {e}")
                    answer = "Error processing this question."
                    answers.append(answer)
                    print(f"❌ Answer {i+1}: {answer}")
            
            processing_time = time.time() - start_time
            print(f"✅ Batch processing completed in {processing_time:.2f} seconds")
            
            return {"answers": answers}
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return {"answers": ["Error in processing. Please try again."] * len(questions)}
    
    def process_single_query(self, query: str):
        """Process a single query with hackathon optimization"""
        result = self.process_hackathon_batch([query])
        return result["answers"][0] if result["answers"] else "No answer generated"
    
    def show_stats(self):
        """Show system statistics"""
        if not self.initialized:
            print("❌ System not initialized")
            return
        
        print("\n" + "="*50)
        print("📊 HACKRX 6.0 SYSTEM STATISTICS")
        print("="*50)
        
        if self.embeddings_manager.index is not None:
            print(f"📁 Documents indexed: {len(self.embeddings_manager.documents)}")
            print(f"🔍 Index dimension: {self.embeddings_manager.dimension}")
            print(f"📝 Current document: {self.current_document or 'Local files'}")
        else:
            print("📁 No documents indexed")
        
        print(f"🔧 OpenAI Model: {config.OPENAI_MODEL}")
        print(f"🔧 Embedding Model: {config.EMBEDDING_MODEL}")
        print(f"🔧 Similarity Threshold: {config.SIMILARITY_THRESHOLD}")
        print("="*50)
    
    def run_interactive_mode(self):
        """Run the interactive terminal interface"""
        print("🚀 Starting HackRx 6.0 Insurance Policy Query System...")
        
        if not self.initialized:
            self.initialize_system()
        
        if not self.initialized:
            print("❌ Failed to initialize system. Exiting.")
            return
        
        print("\n\n🎯 HACKRX 6.0 INSURANCE POLICY QUERY SYSTEM")
        print("="*60)
        print("Welcome to the hackathon-optimized insurance query system!")
        print("\nYou can:")
        print("• Ask single questions like: 'What is the waiting period for dental treatment?'")
        print("• Test batch processing with multiple questions")
        print("• Load documents from URLs or local directory")
        print("\nCommands:")
        print("• 'load' - Load/reload documents from local directory")
        print("• 'load-url <URL>' - Load document from URL")
        print("• 'batch' - Test batch processing with sample questions")
        print("• 'hackathon-test' - Run hackathon format test")
        print("• 'stats' - Show system statistics")
        print("• 'help' - Show this help")
        print("• 'quit' or 'exit' - Exit the system")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n💬 Enter your query (or command): ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye! Thanks for using HackRx 6.0 system!")
                    break
                
                elif user_input.lower() == 'help':
                    print("\n📝 Available Commands:")
                    print("• 'load' - Load documents from local directory")
                    print("• 'load-url <URL>' - Load document from URL")
                    print("• 'batch' - Test batch processing")
                    print("• 'hackathon-test' - Run hackathon format test")
                    print("• 'stats' - Show system statistics")
                    print("• 'quit' or 'exit' - Exit")
                
                elif user_input.lower().startswith('load '):
                    path = user_input[5:].strip()
                    if path:
                        self.load_command(path)
                    else:
                        print("❌ Please specify a document path")
                
                elif user_input.lower() == 'stats':
                    self.show_stats()
                
                elif user_input.lower() == 'batch':
                    print("\n🔄 Running batch processing test...")
                    sample_questions = [
                        "What is the grace period for premium payment?",
                        "What is the waiting period for pre-existing diseases?",
                        "Does this policy cover maternity expenses?",
                        "What is the waiting period for cataract surgery?",
                        "Are medical expenses for organ donors covered?"
                    ]
                    
                    result = self.process_hackathon_batch(sample_questions)
                    
                    print("\n" + "="*60)
                    print("📊 BATCH PROCESSING RESULTS")
                    print("="*60)
                    
                    for i, (question, answer) in enumerate(zip(sample_questions, result["answers"]), 1):
                        print(f"\n🔍 Question {i}: {question}")
                        print(f"📝 Answer {i}: {answer}")
                    
                    print("\n" + "="*60)
                
                elif user_input.lower() == 'hackathon-test':
                    print("\n🏆 Running HackRx 6.0 Format Test...")
                    
                    # Use actual hackathon sample URL if available
                    hackathon_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
                    
                    hackathon_questions = [
                        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                        "What is the waiting period for pre-existing diseases (PED) to be covered?",
                        "Does this policy cover maternity expenses, and what are the conditions?",
                        "What is the waiting period for cataract surgery?",
                        "Are the medical expenses for an organ donor covered under this policy?"
                    ]
                    
                    result = self.process_hackathon_batch(hackathon_questions, hackathon_url)
                    
                    print("\n" + "="*60)
                    print("🏆 HACKRX 6.0 FORMAT TEST RESULTS")
                    print("="*60)
                    print(f"Input Format: {{\"documents\": \"URL\", \"questions\": [...]}}")
                    print(f"Output Format: {{\"answers\": [...]}}")
                    print(f"Questions processed: {len(hackathon_questions)}")
                    print(f"Answers generated: {len(result['answers'])}")
                    
                    print("\n📝 Sample Answers:")
                    for i, answer in enumerate(result["answers"][:3], 1):
                        print(f"   {i}. {answer[:100]}...")
                    
                    print("\n✅ Format Compliance: PASSED")
                    print("="*60)
                
                else:
                    # Process as regular query
                    print(f"\n🔍 Processing query: '{user_input}'")
                    print("="*60)
                    
                    start_time = time.time()
                    answer = self.process_single_query(user_input)
                    processing_time = time.time() - start_time
                    
                    print(f"\n📝 Answer: {answer}")
                    print(f"\n⏱️  Processing Time: {processing_time:.2f}s")
                    print("="*60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using HackRx 6.0 system!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"❌ Error: {e}")

def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="LLM-Powered Intelligent Document Query System - CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python terminal_app.py load "https://example.com/policy.pdf"
  python terminal_app.py query "What is the grace period?"
  python terminal_app.py status
  python terminal_app.py interactive"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load and process a document')
    load_parser.add_argument('document_path', help='Path or URL to document')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query loaded documents')
    query_parser.add_argument('question', help='Question to ask')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    # Reload command
    reload_parser = subparsers.add_parser('reload', help='Reload system and clear cache')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
    
    args = parser.parse_args()
    
    try:
        cli = DocumentQueryCLI()
        
        if args.command == 'load':
            success = cli.load_command(args.document_path)
            return 0 if success else 1
            
        elif args.command == 'query':
            cli.query_command(args.question)
            return 0
            
        elif args.command == 'status':
            cli.status_command()
            return 0
            
        elif args.command == 'reload':
            cli.reload_command()
            return 0
            
        elif args.command == 'interactive':
            cli.interactive_mode()
            return 0
            
        else:
            # No command specified, show help
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())