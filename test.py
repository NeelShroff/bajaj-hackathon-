import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

# To import your DocumentLoader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from document_loader import DocumentLoader

# Set up basic logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_specific_table_chunks(file_path: str, target_page: int = 22):
    """
    Loads a document, processes it, and prints chunks from a specific page's table.
    
    Args:
        file_path (str): The path to the document file to be processed.
        target_page (int): The page number to specifically look for tables on.
    """
    logging.info(f"Starting debugging process for file: {file_path}")
    logging.info(f"Filtering for table chunks on page {target_page}.")
    
    # 1. Initialize the DocumentLoader
    loader = DocumentLoader()
    
    # Check if the file exists
    if not os.path.exists(file_path):
        logging.error(f"File not found at: {file_path}")
        return

    try:
        # 2. Process the document to get all chunks
        all_chunks = loader.process_document(file_path)
        
        if not all_chunks:
            logging.warning("No chunks were generated from the document.")
            return

        logging.info(f"Successfully generated {len(all_chunks)} chunks.")
        
        table_chunks_found = 0
        
        # 3. Iterate through all chunks and filter for the target table
        for i, chunk in enumerate(all_chunks):
            metadata = chunk.metadata
            
            # Check if this chunk originated from a table on the target page
            source = metadata.get('source', '')
            if metadata.get('section_type') == 'table_row' and f"table_{target_page}_" in source:
                table_chunks_found += 1
                logging.info("-" * 50)
                logging.info(f"Chunk Index: {i}")
                logging.info(f"Source: {metadata.get('source', 'N/A')}")
                logging.info(f"Section Type: {metadata.get('section_type', 'N/A')}")
                logging.info(f"Chunk Strategy: {metadata.get('chunk_strategy', 'N/A')}")
                
                # Print the full, processed chunk content
                logging.info("--- Full Chunk Content (for RAG) ---")
                logging.info(chunk.page_content)
                
                # Print the raw row data from metadata for verification
                logging.info("\n--- Raw Row Data (from metadata) ---")
                logging.info(json.dumps(metadata.get('row_data', {}), indent=2, ensure_ascii=False))
                
                logging.info("-" * 50)
        
        if table_chunks_found == 0:
            logging.warning(f"No table chunks were found on page {target_page}.")
        else:
            logging.info(f"Finished debugging. Found {table_chunks_found} table chunks on page {target_page}.")
            
    except Exception as e:
        logging.error(f"An error occurred during chunking: {e}")

if __name__ == '__main__':
    # REPLACE WITH THE PATH TO YOUR ACTUAL POLICY DOCUMENT
    document_path = "data/policies/policy.pdf"
    
    # Run the debugger for the specific document and page
    debug_specific_table_chunks(document_path, target_page=22)
