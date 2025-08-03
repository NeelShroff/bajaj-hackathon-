import os
import logging
from pathlib import Path
from document_loader import DocumentLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("=== Document Loader Tester ===\n")
    
    # Initialize the document loader
    loader = DocumentLoader()
    
    # Get the documents directory from config
    documents_dir = Path("./data/policies")
    
    # Check if directory exists
    if not documents_dir.exists():
        print(f"Error: Documents directory not found at {documents_dir}")
        print("Please create the directory and add some PDF files to test.")
        return
    
    # List available PDF files
    pdf_files = list(documents_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {documents_dir}")
        print("Please add some PDF files to the directory and try again.")
        return
    
    print("Available PDF files:")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"{i}. {pdf_file.name}")
    
    # Let user select a file
    while True:
        try:
            choice = input("\nEnter the number of the PDF to process (or 'q' to quit): ")
            if choice.lower() == 'q':
                return
                
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(pdf_files):
                selected_file = pdf_files[choice_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(pdf_files)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
    
    print(f"\nProcessing file: {selected_file.name}")
    
    try:
        # Process the document
        print("\nExtracting text and creating chunks...")
        chunks = loader.process_document(str(selected_file))
        
        print(f"\nSuccessfully created {len(chunks)} chunks from the document.")
        
        # Show a preview of the chunks
        preview_count = min(3, len(chunks))
        print(f"\nPreview of first {preview_count} chunks:")
        print("-" * 80)
        
        for i, chunk in enumerate(chunks[:preview_count], 1):
            print(f"\nChunk {i} (Section: {chunk.metadata.get('section_id', 'N/A')})")
            print("-" * 40)
            # Show first 200 chars of the chunk
            preview = chunk.page_content[:200]
            if len(chunk.page_content) > 200:
                preview += "..."
            print(preview)
            print(f"\nMetadata: {chunk.metadata}")
            print("-" * 80)
        
        # Ask if user wants to save the processed document
        save_choice = input("\nWould you like to save the processed document? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Enter a name for the processed file (without extension): ")
            if filename:
                loader.save_processed_documents(chunks, filename)
                print(f"Saved processed document as '{filename}.json' in the processed documents directory.")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please check the error message above and try again.")

if __name__ == "__main__":
    main()
