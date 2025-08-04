import os
import json
import logging
import email
import re
import docx
import filetype
import pdfplumber
import PyPDF2
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from openai import OpenAI, APIError
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import io
import pandas as pd

# Try importing specialized libraries, gracefully handle if not installed
try:
    import camelot
    CAMELOT_INSTALLED = True
except ImportError:
    CAMELOT_INSTALLED = False
    logging.warning("Camelot not installed. Table extraction from PDFs may be less effective.")

try:
    from unstructured.partition.auto import partition
    UNSTRUCTURED_INSTALLED = True
except ImportError:
    UNSTRUCTURED_INSTALLED = False
    logging.warning("Unstructured library not installed. Advanced parsing for non-PDFs will be limited.")

try:
    from pdf2image import convert_from_path
    import pytesseract
    PYTESSERACT_INSTALLED = True
except ImportError:
    PYTESSERACT_INSTALLED = False
    logging.warning("pytesseract/pdf2image not installed. OCR for image-based PDFs will not work.")

from config import config

logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    A universal and highly robust DocumentLoader.
    It uses a multi-strategy approach to parse documents of any type, including tables,
    and enriches them with intelligent metadata for superior retrieval.
    """
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.supported_formats = {'.pdf', '.docx', '.doc', '.eml', '.msg', '.txt', '.csv', '.html', '.json', '.xlsx'}
        self.llm_client = OpenAI(api_key=config.OPENAI_API_KEY)

    async def process_document_async(self, file_path: str) -> List[Document]:
        """
        Main async method to process any document type.
        """
        return await asyncio.to_thread(self.process_document, file_path)
    
    def process_document(self, file_path: str) -> List[Document]:
        """
        Main method to process any document type.
        It uses a fallback approach:
        1. Try native format loaders.
        2. If that fails, try a generic unstructured parser.
        3. If tables are detected, they are processed separately.
        """
        logger.info(f"Processing document: {file_path}")
        file_format = self._detect_format(file_path)
        
        text = ""
        tables_data = []

        try:
            if file_format == '.pdf':
                text = self.load_pdf(file_path)
                tables_data = self._extract_tables_from_pdf(file_path)
            elif file_format == '.docx':
                text = self.load_docx(file_path)
            elif file_format in ['.eml', '.msg']:
                text = self.load_email(file_path)
            else:
                text = self._unstructured_parse(file_path)
        except Exception as e:
            logger.error(f"Failed native parsing for {file_path}: {e}. Falling back to unstructured.")
            text = self._unstructured_parse(file_path)

        if not text.strip() and not tables_data:
            raise ValueError(f"No content extracted from {file_path}")

        document_name = Path(file_path).stem
        
        prose_sections = self._llm_based_document_structuring(text) if text else []
        
        all_sections = prose_sections + tables_data
        
        return self.create_chunks(all_sections, document_name)

    def _detect_format(self, file_path: str) -> str:
        """Detects document format, falling back to extension if guessing fails."""
        try:
            kind = filetype.guess(file_path)
            return f".{kind.extension}" if kind else Path(file_path).suffix.lower()
        except Exception:
            return Path(file_path).suffix.lower()

    # --- Text Extraction Methods (Multiple Fallbacks) ---
    def load_pdf(self, file_path: str) -> str:
        text = self._extract_with_pdfplumber(file_path)
        if not text.strip():
            logger.info("pdfplumber failed. Falling back to PyPDF2.")
            text = self._extract_with_pypdf2(file_path)
        if not text.strip() and PYTESSERACT_INSTALLED:
            logger.info("PyPDF2 failed. Falling back to OCR.")
            text = self._extract_with_ocr(file_path)
        return self._clean_text(text)

    def _extract_with_pdfplumber(self, file_path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        return text

    def _extract_with_pypdf2(self, file_path: str) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    if page.extract_text():
                        text += page.extract_text() + "\n"
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
        return text

    def _extract_with_ocr(self, file_path: str) -> str:
        if not PYTESSERACT_INSTALLED:
            return ""
        try:
            images = convert_from_path(file_path)
            return "\n".join([pytesseract.image_to_string(img) for img in images])
        except Exception as e:
            logger.warning(f"OCR failed for {file_path}: {e}")
            return ""

    def load_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    text += "\n" + " | ".join(cells)
        return self._clean_text(text)

    def load_email(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            msg = email.message_from_file(f)
        text = "\n".join(f"{h}: {msg.get(h)}" for h in ["Subject", "From", "To", "Date"] if msg.get(h)) + "\n\n"
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

    def _unstructured_parse(self, file_path: str) -> str:
        if not UNSTRUCTURED_INSTALLED:
            return ""
        try:
            elements = partition(file_path=file_path)
            return self._clean_text("\n\n".join([str(e) for e in elements]))
        except Exception as e:
            logger.warning(f"Unstructured failed for {file_path}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        return ' '.join(text.replace('\x00', '').split())

    # --- Intelligent Structuring and Chunking ---
    def _llm_based_document_structuring(self, text: str) -> List[Dict[str, Any]]:
        """
        Uses an LLM to intelligently parse a document's text into logical sections.
        This is for prose and structured lists, not tables.
        """
        context_limit = 15000
        if len(text) > context_limit:
            text = text[:context_limit] + "..."
            logger.warning("Document text truncated for LLM-based structuring due to length.")

        prompt = f"""
        You are a document parser. Break the following text into logical sections.
        For each section, provide a concise 'section_id', the full 'content', and a 'type' from [definition, benefit, exclusion, condition, general].

        Return only a JSON array of objects.

        Document Text:
        {text}
        
        JSON Response:
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                # Fallback if LLM doesn't return a perfect JSON array
                logger.warning("LLM response did not contain a valid JSON array. Trying to salvage.")
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    raise ValueError("LLM response did not contain a valid JSON array.")
        except (APIError, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"LLM-based structuring failed: {e}. Falling back to default chunking.")
            return [{"section_id": "FULL_DOCUMENT", "content": text, "type": "general"}]

    def _extract_tables_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts tables from PDF files using Camelot for structured data.
        A better, more robust table extraction strategy.
        """
        if not CAMELOT_INSTALLED:
            return []
        
        structured_tables = []
        try:
            tables = camelot.read_pdf(file_path, pages='all', flavor='stream')
            
            for i, table in enumerate(tables):
                # We'll store the table as a pandas DataFrame for granular chunking later
                # A summary of the whole table is also valuable
                summary = self._summarize_table_with_llm(table.df.to_csv(index=False))
                
                structured_tables.append({
                    'section_id': f'table_{i+1}',
                    'content': None,  # We'll chunk the DF itself, so no need for raw content here
                    'type': 'table',
                    'summary': summary,
                    'df': table.df 
                })
        except Exception as e:
            logger.warning(f"Camelot table extraction failed for {file_path}: {e}")
        return structured_tables

    def _summarize_table_with_llm(self, table_text: str) -> str:
        """Uses LLM to create a natural language summary of a table."""
        prompt = f"Summarize the content of the following table text. Be concise, mention headers and key values.\n\nTable:\n{table_text}"
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            return "A table with policy information."

    def create_chunks(self, sections: List[Dict[str, Any]], document_name: str) -> List[Document]:
        """Creates fine-grained chunks from structured sections, including tables."""
        documents = []
        for section in sections:
            section_id = section.get('section_id', 'unknown')
            section_type = section.get('type', 'general')

            if section_type == 'table' and 'df' in section and isinstance(section['df'], pd.DataFrame):
                # For tables, we create a chunk for each row/data point to ensure high recall
                df = section['df']
                headers = df.columns.tolist()
                
                for i, row in df.iterrows():
                    row_data_dict = row.to_dict()
                    
                    # Create a human-readable, semantic-rich chunk from the row
                    # This format is much more searchable than a simple CSV row
                    row_content = "This is from a policy table:\n"
                    for header, value in row_data_dict.items():
                        row_content += f"- {header}: {value}\n"

                    # Add a summary of the whole table for context
                    row_content += f"\nTable Summary: {section.get('summary', 'A table with data.')}"
                    
                    documents.append(Document(page_content=row_content, metadata={
                        'document_name': document_name,
                        'section_id': section_id,
                        'section_type': 'table_row',
                        'chunk_index': i,
                        'source': f"{document_name}_{section_id}_row_{i}",
                        'chunk_strategy': 'table_aware',
                        'row_data': row_data_dict,
                        'table_summary': section.get('summary', 'A table with data.')
                    }))
            else:
                content = section.get('content', '').strip()
                if not content:
                    continue
                # For text, we apply multi-granularity chunking
                small_splitter = RecursiveCharacterTextSplitter(400, 100)
                medium_splitter = RecursiveCharacterTextSplitter(800, 200)

                for variant_type, splitter in {'small': small_splitter, 'medium': medium_splitter}.items():
                    for i, chunk in enumerate(splitter.split_text(content)):
                        documents.append(Document(page_content=chunk, metadata={
                            'document_name': document_name,
                            'section_id': section_id,
                            'section_type': section_type,
                            'chunk_index': i,
                            'source': f"{document_name}_{section_id}_{i}_{variant_type}",
                            'chunk_strategy': f'semantic_{variant_type}'
                        }))
        return documents

    def load_all_documents(self, directory: str = None) -> List[Document]:
        """Loads and processes all documents in parallel."""
        if directory is None:
            directory = config.DOCUMENTS_PATH
        
        all_documents = []
        file_paths = [f for f in Path(directory).glob("*") if f.suffix.lower() in self.supported_formats]

        if not file_paths:
            logger.warning(f"No supported files found in {directory}")
            return all_documents

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            futures = {executor.submit(self.process_document, str(file_path)): file_path for file_path in file_paths}
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"Failed to process {Path(file_path).name}: {e}")
                    continue
        return all_documents