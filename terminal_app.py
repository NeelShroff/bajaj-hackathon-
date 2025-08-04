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
import asyncio

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import config
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingsManager
from src.retrieval import RetrievalSystem
from src.query_processor import QueryProcessor
from src.decision_engine import DecisionEngine

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockResponseFormatter:
    def format_error_response(self, query, error):
        return f"Error: {error}"
    def format_health_response(self, status):
        return status

class DocumentQueryCLI:
    """Command-line interface for the Document Query System"""
    
    def __init__(self):
        self.document_loader = None
        self.embeddings_manager = None
        self.retrieval_system = None
        self.query_processor = None
        self.decision_engine = None
        self.response_formatter = MockResponseFormatter()
        self.initialized = False
        self.loaded_documents = {}
        self.query_history = []
        self.current_document = None
    
    def initialize_system(self):
        try:
            print("🔄 Initializing Document Query System...")
            config.validate()
            print("✅ Configuration validated")

            print("🔄 Loading document processor...")
            self.document_loader = DocumentLoader()

            print("🔄 Loading embeddings manager...")
            self.embeddings_manager = EmbeddingsManager()

            print("🔄 Loading retrieval system...")
            self.retrieval_system = RetrievalSystem()

            print("🔄 Loading query processor...")
            self.query_processor = QueryProcessor()

            print("🔄 Loading decision engine...")
            self.decision_engine = DecisionEngine()

            self.initialized = True
            print("✅ System initialization completed!")

        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            print(f"❌ Initialization failed: {e}")
            self.initialized = False

    async def load_command(self, document_path: str):
        if not self.initialized:
            self.initialize_system()

        if not self.initialized:
            print("❌ Failed to initialize system")
            return False

        try:
            print(f"🔄 Loading document: {document_path}")
            start_time = time.time()

            if document_path.startswith('http'):
                print(f"📥 Downloading from URL: {document_path}")
                response = requests.get(document_path, timeout=60)
                response.raise_for_status()

                content_type = response.headers.get('content-type', '')
                if 'pdf' in content_type:
                    suffix = '.pdf'
                elif 'word' in content_type or 'docx' in content_type:
                    suffix = '.docx'
                else:
                    suffix = '.pdf'

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(response.content)
                    temp_path = temp_file.name

                documents = await asyncio.to_thread(self.document_loader.process_document, temp_path)
                os.unlink(temp_path)

            else:
                if not os.path.exists(document_path):
                    print(f"❌ File not found: {document_path}")
                    return False

                documents = await asyncio.to_thread(self.document_loader.process_document, document_path)

            # ✅ FIX: Unwrap if wrapped in dict
            if isinstance(documents, dict) and 'chunks' in documents:
                documents = documents['chunks']

            if not documents:
                print("❌ No content extracted from document")
                return False

            print(f"🔄 Creating embeddings for {len(documents)} chunks and upserting to Pinecone...")
            await self.embeddings_manager.build_index_async(documents)

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
            print(f"📄 Format: {Path(document_path).suffix}")
            print(f"📏 Total chunks: {len(documents)}")
            print("💾 Index updated on Pinecone.")
            return True

        except Exception as e:
            logger.error(f"Error loading document: {e}")
            print(f"❌ Error loading document: {e}")
            return False

    async def query_command(self, question: str):
        if not self.initialized:
            self.initialize_system()
            if not self.initialized:
                print("❌ System initialization failed.")
                return

        try:
            print(f"🔍 Processing query: {question}")
            start_time = time.time()

            answer, sources, confidence = await self.retrieval_system.retrieve_and_generate_answer_async(
                query=question,
                embeddings_manager=self.embeddings_manager
            )

            processing_time = time.time() - start_time

            print(f"\n📋 Answer:")
            print(f"{answer}")
            print(f"\n⏱️ Processing time: {processing_time:.2f}s")
            print(f"🔍 Retrieved {len(sources)} relevant chunks")

            print(f"\n📚 Sources:")
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source}")

            self.query_history.append({
                'question': question,
                'answer': answer,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'chunks_retrieved': len(sources)
            })

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"❌ Error processing query: {e}")

    def status_command(self, run_async_tasks=False):
        if not self.initialized:
            self.initialize_system()

        print("\n📊 System Status")
        print("=" * 50)
        print(f"🔧 Initialized: {'✅ Yes' if self.initialized else '❌ No'}")

        if run_async_tasks:
            index_stats = asyncio.run(asyncio.to_thread(self.embeddings_manager.get_index_stats))
        else:
            index_stats = self.embeddings_manager.get_index_stats()

        index_status = index_stats.get('status')
        total_vectors = index_stats.get('total_vectors', 'N/A')

        print(f"📁 Documents indexed (Pinecone): {'✅ Ready' if index_status == 'ready' else '❌ Not ready'}")
        print(f"🔍 Vector count: {total_vectors}")
        print(f"📝 Query history: {len(self.query_history)} queries")
        print(f"⚙️ Config: {config.OPENAI_MODEL} | {config.EMBEDDING_MODEL}")

        if self.query_history:
            print(f"\n📈 Recent Queries ({len(self.query_history)} total):")
            for i, query in enumerate(self.query_history[-3:], 1):
                print(f"  {i}. {query['question'][:60]}{'...' if len(query['question']) > 60 else ''}")
                print(f"    ⏱️ {query['processing_time']:.2f}s | 🔍 {query['chunks_retrieved']} chunks")

    def reload_command(self):
        print("🔄 Reloading system...")
        self.initialized = False
        self.initialize_system()
        if self.initialized:
            print("✅ System reloaded successfully")
        else:
            print("❌ Failed to reload system")

    def interactive_mode(self):
        print("💬 Starting Interactive Mode")
        print("Type 'help' for commands, 'exit' to quit\n")

        if not self.initialized:
            self.initialize_system()

        if self.embeddings_manager.get_index_stats().get('total_vectors', 0) == 0:
            print("⚠️ No documents loaded in Pinecone. Use 'load <path>' to load a document first.")

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
                    print("  load <path>     - Load a document to Pinecone (can take a while)")
                    print("  status          - Show system status and Pinecone index stats")
                    print("  reload          - Reload system")
                    print("  help            - Show this help")
                    print("  exit/quit/q     - Exit interactive mode")
                    print("  <question>      - Ask a question about loaded documents")

                elif user_input.lower().startswith('load '):
                    path = user_input[5:].strip()
                    if path:
                        asyncio.run(self.load_command(path))
                    else:
                        print("❌ Please specify a document path")

                elif user_input.lower() == 'status':
                    self.status_command()

                elif user_input.lower() == 'reload':
                    self.reload_command()

                else:
                    asyncio.run(self.query_command(user_input))

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"❌ Error: {e}")

def main():
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

    load_parser = subparsers.add_parser('load', help='Load and process a document')
    load_parser.add_argument('document_path', help='Path or URL to document')

    query_parser = subparsers.add_parser('query', help='Query loaded documents')
    query_parser.add_argument('question', help='Question to ask')

    subparsers.add_parser('status', help='Show system status')
    subparsers.add_parser('reload', help='Reload system and clear cache')
    subparsers.add_parser('interactive', help='Start interactive mode')

    args = parser.parse_args()

    try:
        cli = DocumentQueryCLI()

        if args.command == 'load':
            success = asyncio.run(cli.load_command(args.document_path))
            return 0 if success else 1

        elif args.command == 'query':
            asyncio.run(cli.query_command(args.question))
            return 0

        elif args.command == 'status':
            cli.status_command(run_async_tasks=True)
            return 0

        elif args.command == 'reload':
            cli.reload_command()
            return 0

        elif args.command == 'interactive':
            cli.interactive_mode()
            return 0

        else:
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
