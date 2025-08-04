import os
import json
import logging
import email
import mimetypes
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2
import pdfplumber
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from datetime import datetime

from config import config

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handles loading and processing of PDF, DOCX, and email documents."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.supported_formats = {'.pdf', '.docx', '.doc', '.eml', '.msg'}
        
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
        """Create LangChain Document chunks from sections with UNIVERSAL document handling."""
        documents = []
        
        for section in sections:
            # STRATEGY 1: Standard chunking
            section_chunks = self.text_splitter.split_text(section['content'])
            
            for i, chunk in enumerate(section_chunks):
                metadata = {
                    'document_name': document_name,
                    'section_id': section['section_id'],
                    'section_type': section['type'],
                    'chunk_index': i,
                    'total_chunks': len(section_chunks),
                    'source': f"{document_name}_{section['section_id']}_{i}",
                    'chunk_strategy': 'standard'
                }
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=metadata
                ))
            
            # STRATEGY 2: Definition-aware chunking for terms like "Grace Period means..."
            content = section['content']
            definition_patterns = [
                r'(\d+\.\d+\s+[A-Z][^.]*?means[^.]*?\.(?:[^.]*?\.)*)',  # "2.21 Grace Period means..."
                r'([A-Z][^.]*?means[^.]*?\.(?:[^.]*?\.)*)',  # "Grace Period means..."
                r'(\"[^\"]*\"[^.]*?\.(?:[^.]*?\.)*)',  # Quoted definitions
                r'([A-Z][A-Z\s]+:[^.]*?\.(?:[^.]*?\.)*)'  # "GRACE PERIOD: ..."
            ]
            
            for pattern in definition_patterns:
                import re
                matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    if len(match.strip()) > 50:  # Only meaningful definitions
                        metadata = {
                            'document_name': document_name,
                            'section_id': section['section_id'],
                            'section_type': 'definition',
                            'chunk_index': f"def_{len(documents)}",
                            'total_chunks': len(section_chunks),
                            'source': f"{document_name}_{section['section_id']}_def",
                            'chunk_strategy': 'definition_aware'
                        }
                        
                        documents.append(Document(
                            page_content=match.strip(),
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
    
    def load_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(file_path)
            text = ""
            
            # Extract text from paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text += " | ".join(row_text) + "\n"
            
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            raise
    
    def load_email(self, file_path: str) -> str:
        """Extract text from email file (.eml)."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                msg = email.message_from_file(f)
            
            text = ""
            
            # Extract headers
            headers = ["Subject", "From", "To", "Date"]
            for header in headers:
                value = msg.get(header)
                if value:
                    text += f"{header}: {value}\n"
            
            text += "\n" + "-" * 50 + "\n\n"
            
            # Extract body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            text += payload.decode('utf-8', errors='ignore')
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    text += payload.decode('utf-8', errors='ignore')
            
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from email {file_path}: {e}")
            raise
    
    def _detect_format(self, file_path: str) -> str:
        """Detect document format from file extension."""
        ext = Path(file_path).suffix.lower()
        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: {self.supported_formats}")
        return ext
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load and process a document of any supported format, returning chunked documents."""
        try:
            logger.info(f"Loading document: {file_path}")
            
            # Detect format and extract text
            file_format = self._detect_format(file_path)
            
            if file_format == '.pdf':
                text = self.load_pdf(file_path)
            elif file_format in ['.docx', '.doc']:
                text = self.load_docx(file_path)
            elif file_format in ['.eml', '.msg']:
                text = self.load_email(file_path)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            if not text.strip():
                raise ValueError(f"No text extracted from {file_path}")
            
            # Create enhanced metadata
            metadata = {
                "source": file_path,
                "filename": os.path.basename(file_path),
                "file_format": file_format,
                "file_size": os.path.getsize(file_path),
                "total_chars": len(text),
                "processed_at": datetime.now().isoformat(),
                "loader_version": "2.0"
            }
            
            # Split into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create Document objects with enhanced metadata
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "chunk_start_char": sum(len(c) for c in chunks[:i]),
                    "chunk_end_char": sum(len(c) for c in chunks[:i+1])
                })
                documents.append(Document(page_content=chunk, metadata=chunk_metadata))
            
            logger.info(f"Successfully loaded {len(documents)} chunks from {file_path} ({file_format})")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
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