#!/usr/bin/env python3
"""
Terminal-based Insurance Policy Query System
Runs the complete system functionality in terminal with print/input interface
"""

import logging
import os
import sys
import json
from typing import Optional, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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

class TerminalInsuranceSystem:
    """Terminal-based insurance policy query system"""
    
    def __init__(self):
        self.document_loader = None
        self.embeddings_manager = None
        self.query_processor = None
        self.retrieval_system = None
        self.decision_engine = None
        self.response_formatter = None
        self.initialized = False
    
    def initialize_system(self):
        """Initialize all system components"""
        try:
            print("🔄 Initializing Insurance Policy Query System...")
            
            # Validate configuration
            config.validate()
            print("✅ Configuration validated")
            
            # Initialize components
            print("🔄 Loading document processor...")
            self.document_loader = DocumentLoader()
            
            print("🔄 Loading embeddings manager...")
            self.embeddings_manager = EmbeddingsManager()
            
            print("🔄 Loading query processor...")
            self.query_processor = QueryProcessor()
            
            print("🔄 Loading retrieval system...")
            self.retrieval_system = RetrievalSystem(self.embeddings_manager)
            
            print("🔄 Loading decision engine...")
            self.decision_engine = DecisionEngine()
            
            print("🔄 Loading response formatter...")
            self.response_formatter = ResponseFormatter()
            
            # Try to load existing index
            if self.embeddings_manager.load_index():
                print("✅ Loaded existing FAISS index")
            else:
                print("⚠️  No existing index found. You'll need to load documents first.")
            
            self.initialized = True
            print("✅ System initialization completed!\n")
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            logger.error(f"System initialization failed: {e}")
            return False
        
        return True
    
    def load_documents(self):
        """Load and index documents from the data/policies directory"""
        try:
            print("🔄 Loading documents from data/policies directory...")
            
            # Load all documents
            documents = self.document_loader.load_all_documents()
            
            if not documents:
                print("❌ No documents found in data/policies directory")
                return False
            
            print(f"✅ Found {len(documents)} document chunks")
            
            # Build index
            print("🔄 Building FAISS index...")
            self.embeddings_manager.build_index(documents)
            
            # Save index
            print("🔄 Saving index...")
            self.embeddings_manager.save_index()
            
            print(f"✅ Successfully indexed {len(documents)} document chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            logger.error(f"Error loading documents: {e}")
            return False
    
    def process_single_query(self, query: str):
        """Process a single query and display results"""
        try:
            print(f"\n🔍 Processing query: '{query}'")
            print("=" * 60)
            
            # Process query
            processed_query = self.query_processor.process_query(query)
            print("✅ Query processed and entities extracted")
            
            # Retrieve relevant documents
            retrieval_results = self.retrieval_system.contextual_retrieval(processed_query)
            
            # Count total documents from all categories
            total_docs = 0
            for category, docs in retrieval_results.items():
                total_docs += len(docs)
            print(f"✅ Retrieved {total_docs} relevant document chunks across {len(retrieval_results)} categories")
            
            # Make decision
            decision_result = self.decision_engine.evaluate_coverage(
                processed_query, retrieval_results
            )
            print("✅ Decision analysis completed")
            
            # Format response
            formatted_response = self.response_formatter.format_response(
                query, processed_query, decision_result, retrieval_results
            )
            print("✅ Response formatted")
            
            # Display results
            self.display_results(formatted_response)
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            logger.error(f"Error processing query: {e}")
    
    def display_results(self, response: dict):
        """Display formatted results to terminal"""
        print("\n" + "="*60)
        print("📋 QUERY RESULTS")
        print("="*60)
        
        # Basic info
        print(f"🔍 Original Query: {response.get('query', 'N/A')}")
        
        # Extracted entities from query_analysis
        query_analysis = response.get('query_analysis', {})
        entities = query_analysis.get('extracted_entities', {})
        if entities:
            print("\n👤 Extracted Information:")
            for key, value in entities.items():
                if value:
                    print(f"   • {key.replace('_', ' ').title()}: {value}")
        
        # Decision (decision is a string, not a dict)
        decision = response.get('decision', 'Unknown')
        confidence = response.get('confidence', 0.0)
        print(f"\n🎯 Coverage Decision: {decision}")
        print(f"📊 Confidence: {confidence:.1%}")
        
        # Amount information
        amount = response.get('amount', {})
        if amount:
            covered_amount = amount.get('covered_amount', 0)
            currency = amount.get('currency', 'INR')
            print(f"💰 Covered Amount: {covered_amount:,} {currency}")
            
            patient_resp = amount.get('patient_responsibility', 0)
            if patient_resp > 0:
                print(f"💳 Patient Responsibility: {patient_resp:,} {currency}")
        
        # Justification
        justification = response.get('justification', {})
        if justification:
            reasoning = justification.get('reasoning', 'No reasoning provided')
            print(f"\n📝 Reasoning:")
            print(f"   {reasoning}")
            
            # Applicable clauses
            clauses = justification.get('applicable_clauses', [])
            if clauses:
                print(f"\n📚 Applicable Policy Clauses:")
                for i, clause in enumerate(clauses[:3], 1):  # Show top 3
                    clause_text = clause.get('content', str(clause))[:150] + "..." if len(str(clause)) > 150 else str(clause)
                    print(f"   {i}. {clause_text}")
            
            # Conditions
            conditions = justification.get('conditions', [])
            if conditions:
                print(f"\n⚠️  Conditions:")
                for condition in conditions:
                    print(f"   • {condition}")
            
            # Exclusions checked
            exclusions = justification.get('exclusions_checked', [])
            if exclusions:
                print(f"\n❌ Exclusions Checked:")
                for exclusion in exclusions:
                    print(f"   • {exclusion}")
        
        # Metadata
        metadata = response.get('metadata', {})
        if metadata:
            processing_time = metadata.get('processing_time', 'Unknown')
            print(f"\n⏱️  Processing Time: {processing_time}")
        
        print("\n" + "="*60)
    
    def show_system_stats(self):
        """Display system statistics"""
        try:
            print("\n📊 SYSTEM STATISTICS")
            print("="*40)
            
            # Index stats
            if self.embeddings_manager:
                stats = self.embeddings_manager.get_index_stats()
                print(f"📚 Documents indexed: {stats.get('total_documents', 0)}")
                print(f"🔍 Index status: {stats.get('status', 'Unknown')}")
                print(f"📏 Vector dimension: {stats.get('dimension', 'Unknown')}")
            
            # Configuration
            print(f"\n⚙️  Configuration:")
            print(f"   • Chunk size: {config.CHUNK_SIZE}")
            print(f"   • Chunk overlap: {config.CHUNK_OVERLAP}")
            print(f"   • Top K results: {config.TOP_K_RESULTS}")
            print(f"   • Similarity threshold: {config.SIMILARITY_THRESHOLD}")
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
    
    def run_interactive_mode(self):
        """Run the interactive terminal interface"""
        print("\n🎯 INSURANCE POLICY QUERY SYSTEM")
        print("="*50)
        print("Welcome to the terminal-based insurance query system!")
        print("You can ask questions like:")
        print("• '46M, knee surgery, Pune, 3-month policy'")
        print("• 'Does my policy cover heart surgery for 35-year-old female?'")
        print("• 'What is the waiting period for dental treatment?'")
        print("\nCommands:")
        print("• 'load' - Load/reload documents")
        print("• 'stats' - Show system statistics") 
        print("• 'help' - Show this help")
        print("• 'quit' or 'exit' - Exit the system")
        print("="*50)
        
        while True:
            try:
                query = input("\n💬 Enter your query (or command): ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif query.lower() == 'help':
                    print("\n📖 Available commands:")
                    print("• 'load' - Load/reload documents from data/policies")
                    print("• 'stats' - Show system statistics")
                    print("• 'help' - Show this help")
                    print("• 'quit' or 'exit' - Exit the system")
                    print("\nOr ask any insurance-related question!")
                
                elif query.lower() == 'load':
                    self.load_documents()
                
                elif query.lower() == 'stats':
                    self.show_system_stats()
                
                else:
                    # Process as insurance query
                    if not self.initialized:
                        print("❌ System not initialized. Please restart.")
                        continue
                    
                    if not self.embeddings_manager.documents:
                        print("⚠️  No documents loaded. Use 'load' command to load documents first.")
                        continue
                    
                    self.process_single_query(query)
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Interactive mode error: {e}")

def main():
    """Main entry point"""
    print("🚀 Starting Insurance Policy Query System...")
    
    # Create system instance
    system = TerminalInsuranceSystem()
    
    # Initialize system
    if not system.initialize_system():
        print("❌ Failed to initialize system. Exiting.")
        return 1
    
    # Run interactive mode
    system.run_interactive_mode()
    
    return 0

if __name__ == "__main__":
    exit(main())
