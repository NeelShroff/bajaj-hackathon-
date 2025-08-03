import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import config

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handles loading and processing of PDF policy documents."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def load_pdf(self, file_path: str) -> str:
        """Extract text from PDF file using multiple methods for better accuracy."""
        try:
            # Try pdfplumber first for better text extraction
            text = self._extract_with_pdfplumber(file_path)
            if not text.strip():
                # Fallback to PyPDF2
                text = self._extract_with_pypdf2(file_path)
            
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
    
    def _extract_with_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber."""
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def _extract_with_pypdf2(self, file_path: str) -> str:
        """Extract text using PyPDF2 as fallback."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Remove special characters that might interfere with processing
        text = text.replace('\x00', '')
        return text
    
    def extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract structured sections from policy document."""
        sections = []
        current_section = None
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect various section headers patterns
            line_upper = line.upper()
            is_section_header = False
            
            # Pattern 1: "SECTION A", "SECTION B", etc.
            if line_upper.startswith('SECTION ') and len(line) < 50:
                is_section_header = True
            # Pattern 2: "SECTION A)", "SECTION B)", etc.
            elif line_upper.startswith('SECTION ') and ')' in line and len(line) < 100:
                is_section_header = True
            # Pattern 3: All caps headers that look like section titles
            elif (len(line) < 100 and line.isupper() and 
                  any(keyword in line_upper for keyword in ['COVERAGE', 'BENEFITS', 'EXCLUSIONS', 'CONDITIONS', 'TERMS', 'CLAIMS', 'DEFINITIONS'])):
                is_section_header = True
                
            if is_section_header:
                # Save previous section if exists
                if current_section and current_content:
                    sections.append({
                        'section_id': current_section,
                        'content': '\n'.join(current_content),
                        'type': 'policy_section'
                    })
                
                current_section = line
                current_content = []
            else:
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            sections.append({
                'section_id': current_section,
                'content': '\n'.join(current_content),
                'type': 'policy_section'
            })
        
        # If no sections found, create a single section with all content
        if not sections and text.strip():
            sections.append({
                'section_id': 'FULL_DOCUMENT',
                'content': text.strip(),
                'type': 'full_document'
            })
        
        return sections
    
    def create_chunks(self, sections: List[Dict[str, Any]], document_name: str) -> List[Document]:
        """Create LangChain Document chunks from sections."""
        documents = []
        
        for section in sections:
            # Split section content into chunks
            section_chunks = self.text_splitter.split_text(section['content'])
            
            for i, chunk in enumerate(section_chunks):
                metadata = {
                    'document_name': document_name,
                    'section_id': section['section_id'],
                    'section_type': section['type'],
                    'chunk_index': i,
                    'total_chunks': len(section_chunks),
                    'source': f"{document_name}_{section['section_id']}_{i}"
                }
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=metadata
                ))
        
        return documents
    
    def process_document(self, file_path: str) -> List[Document]:
        """Process a single document and return chunks."""
        logger.info(f"Processing document: {file_path}")
        
        # Extract text
        text = self.load_pdf(file_path)
        
        # Extract sections
        sections = self.extract_sections(text)
        
        # Create chunks
        document_name = Path(file_path).stem
        chunks = self.create_chunks(sections, document_name)
        
        logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections")
        return chunks
    
    def load_all_documents(self, directory: str = None) -> List[Document]:
        """Load and process all PDF documents in the specified directory."""
        if directory is None:
            directory = config.DOCUMENTS_PATH
        
        all_documents = []
        pdf_files = list(Path(directory).glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return all_documents
        
        for pdf_file in pdf_files:
            try:
                documents = self.process_document(str(pdf_file))
                all_documents.extend(documents)
                logger.info(f"Processed {pdf_file.name}: {len(documents)} chunks")
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                continue
        
        return all_documents
    
    def save_processed_documents(self, documents: List[Document], filename: str):
        """Save processed documents to cache."""
        cache_path = Path(config.PROCESSED_DOCS_PATH)
        cache_path.mkdir(exist_ok=True)
        
        file_path = cache_path / f"{filename}.json"
        
        # Convert documents to serializable format
        serializable_docs = []
        for doc in documents:
            serializable_docs.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_docs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(documents)} documents to {file_path}")
    
    def load_processed_documents(self, filename: str) -> List[Document]:
        """Load processed documents from cache."""
        cache_path = Path(config.PROCESSED_DOCS_PATH) / f"{filename}.json"
        
        if not cache_path.exists():
            logger.warning(f"Cache file not found: {cache_path}")
            return []
        
        with open(cache_path, 'r', encoding='utf-8') as f:
            serializable_docs = json.load(f)
        
        documents = []
        for doc_data in serializable_docs:
            documents.append(Document(
                page_content=doc_data['page_content'],
                metadata=doc_data['metadata']
            ))
        
        logger.info(f"Loaded {len(documents)} documents from cache")
        return documents 