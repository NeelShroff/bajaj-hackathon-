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

# Dummy config for demonstration, replace with your actual config
class Config:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    OPENAI_API_KEY = "YOUR_API_KEY"  # <-- REMINDER: FIX THIS API KEY!
    DOCUMENTS_PATH = "./documents"

config = Config()

logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    A universal and highly robust DocumentLoader.
    It uses a multi-strategy approach to parse documents of any type, including tables,
    and enriches them with intelligent metadata for superior retrieval.
    """
    def __init__(self):
        # Enhanced chunking strategy for policy documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # Increased to preserve complete definitions
            chunk_overlap=300,  # Increased overlap to maintain context
            separators=[
                "\n\n\n",  # Major section breaks
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentence breaks
                "; ",      # Clause breaks
                ", ",      # Phrase breaks
                " ",       # Word breaks
                ""
            ]
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

    def _llm_based_document_structuring(self, text: str) -> List[Dict[str, Any]]:
        # Increased context limit to preserve more document content
        context_limit = 40000  # Increased from 15000 to 40000 for better coverage
        
        if len(text) > context_limit:
            # For very large documents, use simpler chunking to avoid losing content
            logger.info(f"Large document ({len(text)} chars), using enhanced chunking instead of LLM structuring")
            return self._create_enhanced_chunks_from_text(text)
        
        # For smaller documents, use LLM structuring but with no truncation

        prompt = f"""
        You are a document parser. Break the following text into logical sections.
        For each section, provide a concise 'section_id', the full 'content', and a 'type' from [definition, benefit, exclusion, condition, general].
        Return only a JSON array of objects.
        Document Text:
        {text}
        JSON Response:
        """
        
        try:
            if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "YOUR_API_KEY":
                raise ValueError("API key not set or is a placeholder.")

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
                logger.warning("LLM response did not contain a valid JSON array. Trying to salvage.")
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    raise ValueError("LLM response did not contain a valid JSON array.")
        except (APIError, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"LLM-based structuring failed: {e}. Falling back to default chunking.")
            return [{"section_id": "FULL_DOCUMENT", "content": text, "type": "general"}]
    
    def _create_enhanced_chunks_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Create enhanced chunks from large documents without losing content.
        Uses semantic boundaries and preserves complete sections.
        """
        # Split by major section indicators first
        major_sections = []
        current_section = ""
        
        lines = text.split('\n')
        section_indicators = ['SECTION', 'CHAPTER', 'PART', 'ARTICLE', 'CLAUSE', 'DEFINITION', 'BENEFIT', 'EXCLUSION', 'CONDITION']
        
        for line in lines:
            line_upper = line.upper().strip()
            is_section_start = any(indicator in line_upper for indicator in section_indicators)
            
            if is_section_start and current_section.strip():
                # Save previous section
                major_sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        # Add the last section
        if current_section.strip():
            major_sections.append(current_section.strip())
        
        # If no major sections found, split by paragraphs
        if len(major_sections) <= 1:
            major_sections = [section.strip() for section in text.split('\n\n') if section.strip()]
        
        # Convert to document sections
        sections = []
        for i, section_text in enumerate(major_sections):
            if len(section_text) > 100:  # Only include substantial sections
                sections.append({
                    "section_id": f"SECTION_{i+1}",
                    "content": section_text,
                    "type": "general"
                })
        
        return sections if sections else [{"section_id": "FULL_DOCUMENT", "content": text, "type": "general"}]

    def _extract_tables_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        tables = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    extracted_tables = page.extract_tables()
                    for i, table_data in enumerate(extracted_tables):
                        if not table_data or not table_data[0]:
                            continue
                            
                        headers_rows = [row for row in table_data if any(str(c).strip() for c in row)][:2]
                        data_start_row_index = len(headers_rows)
                        
                        if len(headers_rows) > 1:
                            first_row = [str(c).strip() if c else "" for c in headers_rows[0]]
                            second_row = [str(c).strip() if c else "" for c in headers_rows[1]]
                            
                            merged_headers = []
                            last_main_header = None
                            for j in range(len(first_row)):
                                current_main_header = first_row[j] if j < len(first_row) and first_row[j] else ''
                                current_sub_header = second_row[j] if j < len(second_row) and second_row[j] else ''
                                
                                if current_main_header:
                                    last_main_header = current_main_header
                                    if current_sub_header:
                                        merged_headers.append(f"{last_main_header} - {current_sub_header}")
                                    else:
                                        merged_headers.append(last_main_header)
                                elif current_sub_header:
                                    merged_headers.append(f"{last_main_header} - {current_sub_header}")
                                else:
                                    merged_headers.append(f"Col_{len(merged_headers)}")
                            
                            headers = merged_headers
                            data_start_row_index = len(headers_rows)
                        else:
                            headers = [str(cell).strip() if cell else f"Col_{j}" for j, cell in enumerate(headers_rows[0])]
                            data_start_row_index = 1
                            
                        data_rows = table_data[data_start_row_index:]
                        
                        if not headers or not data_rows:
                            continue
                        
                        rows_list = []
                        for row in data_rows:
                            row_dict = {}
                            for j, header in enumerate(headers):
                                cell_value = str(row[j]).strip() if j < len(row) and row[j] else ""
                                row_dict[header] = cell_value
                            
                            if any(v for v in row_dict.values()):
                                rows_list.append(row_dict)

                        if rows_list:
                            tables.append({
                                'section_id': f'table_{page_num+1}_{i+1}',
                                'content': rows_list,
                                'type': 'table'
                            })
        except Exception as e:
            logger.warning(f"pdfplumber table extraction failed for {file_path}: {e}")
        return tables

    def create_chunks(self, sections: List[Dict[str, Any]], document_name: str) -> List[Document]:
        documents = []
        for section in sections:
            section_id = section.get('section_id', 'unknown')
            section_type = section.get('type', 'general')

            if section_type == 'table' and 'content' in section and isinstance(section['content'], list):
                table_rows = section['content']
                for i, row_data in enumerate(table_rows):
                    # Generic table-to-prose conversion for any table structure
                    if not row_data or not isinstance(row_data, dict):
                        continue
                    
                    # Find the primary identifier column (first non-empty column)
                    primary_key = None
                    primary_value = None
                    
                    for key, value in row_data.items():
                        if value and str(value).strip():
                            primary_key = key
                            primary_value = str(value).strip()
                            break
                    
                    if not primary_key or not primary_value:
                        continue
                    
                    # Collect all other meaningful data columns
                    data_pairs = []
                    for key, value in row_data.items():
                        if key != primary_key and value and str(value).strip():
                            clean_key = self._clean_column_name(key)
                            clean_value = str(value).strip()
                            if clean_value.lower() not in ['', 'na', 'n/a', 'null', 'none']:
                                data_pairs.append((clean_key, clean_value))
                    
                    # Skip if no meaningful data beyond the primary identifier
                    if not data_pairs:
                        continue
                    
                    # Clean the primary identifier
                    clean_primary = self._clean_text_content(primary_value)
                    if not clean_primary:
                        continue
                    
                    # Generate natural language content
                    row_content = self._generate_table_prose(clean_primary, data_pairs)
                    
                    if not row_content:
                        continue

                    documents.append(Document(page_content=row_content, metadata={
                        'document_name': document_name,
                        'section_id': section_id,
                        'section_type': 'table_row',
                        'chunk_index': i,
                        'source': f"{document_name}_{section_id}_row_{i}",
                        'chunk_strategy': 'table_prose',
                        'row_data': json.dumps(row_data),
                        'primary_key': primary_key,
                        'data_columns': len(data_pairs)
                    }))
            else:
                content = section.get('content', '').strip()
                if not content:
                    continue
                
                # Enhanced chunking with definition preservation
                chunks = self._smart_chunk_with_definitions(content)

                for i, chunk in enumerate(chunks):
                    documents.append(Document(page_content=chunk, metadata={
                        'document_name': document_name,
                        'section_id': section_id,
                        'section_type': section_type,
                        'chunk_index': i,
                        'source': f"{document_name}_{section_id}_{i}",
                        'chunk_strategy': 'definition_aware'
                    }))
        return documents

    def _smart_chunk_with_definitions(self, content: str) -> List[str]:
        """Smart chunking that preserves complete policy definitions and key facts."""
        if not content.strip():
            return []
        
        # First, identify and preserve complete definitions
        definition_patterns = [
            r'(\d+\.\d+\s+[A-Z][^\n]*?\s+means\s+[^\n]*?(?:\n[^\n]*?)*?)(?=\n\n|\n\d+\.\d+|$)',
            r'([A-Z][^\n]*?\s+means\s+[^\n]*?(?:\n[^\n]*?)*?)(?=\n\n|\n[A-Z][^\n]*?\s+means|$)',
            r'(\"[^\"]+\"\s+means\s+[^\n]*?(?:\n[^\n]*?)*?)(?=\n\n|\n\"[^\"]+\"|$)',
            r'([A-Z][A-Z\s]+:\s*[^\n]*?(?:\n[^\n]*?)*?)(?=\n\n|\n[A-Z][A-Z\s]+:|$)'
        ]
        
        preserved_definitions = []
        remaining_content = content
        
        # Extract complete definitions first
        for pattern in definition_patterns:
            matches = re.finditer(pattern, remaining_content, re.MULTILINE | re.DOTALL)
            for match in matches:
                definition_text = match.group(1).strip()
                if len(definition_text) > 50:  # Only preserve substantial definitions
                    preserved_definitions.append(definition_text)
                    # Remove from remaining content to avoid duplication
                    remaining_content = remaining_content.replace(match.group(0), '', 1)
        
        # Clean up remaining content
        remaining_content = re.sub(r'\n\s*\n\s*\n', '\n\n', remaining_content).strip()
        
        # Chunk the remaining content normally
        regular_chunks = self.text_splitter.split_text(remaining_content) if remaining_content else []
        
        # Combine preserved definitions with regular chunks
        all_chunks = preserved_definitions + regular_chunks
        
        # Filter out empty or very short chunks
        final_chunks = [chunk.strip() for chunk in all_chunks if chunk.strip() and len(chunk.strip()) > 20]
        
        # If no chunks were created, fall back to regular splitting
        if not final_chunks:
            final_chunks = self.text_splitter.split_text(content)
        
        return final_chunks

    def _clean_column_name(self, column_name: str) -> str:
        """Clean and normalize column names for better readability."""
        if not column_name:
            return ''
        
        # Remove common prefixes and suffixes
        clean_name = str(column_name).strip()
        
        # Remove special characters and normalize
        clean_name = re.sub(r'[*#@$%^&()\[\]{}|\\:;"<>?/~`]', '', clean_name)
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        
        # Remove common table formatting artifacts
        clean_name = re.sub(r'^(Plans?\s*-\s*|Plan\s*-\s*)', '', clean_name, flags=re.IGNORECASE)
        clean_name = re.sub(r'\s*(Column|Col)\s*\d+', '', clean_name, flags=re.IGNORECASE)
        
        return clean_name.strip()
    
    def _clean_text_content(self, text: str) -> str:
        """Clean and normalize text content for better readability."""
        if not text:
            return ''
        
        clean_text = str(text).strip()
        
        # Remove excessive punctuation and formatting
        clean_text = re.sub(r'[*#@$%^&()\[\]{}|\\:;"<>?/~`]+', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Remove parenthetical notes that might be domain-specific
        clean_text = re.sub(r'\s*\([^)]*\)\s*', ' ', clean_text).strip()
        
        return clean_text
    
    def _generate_table_prose(self, primary_item: str, data_pairs: List[tuple]) -> str:
        """Generate natural language prose from table row data."""
        if not primary_item or not data_pairs:
            return ''
        
        # Check if all values are the same (common case)
        values = [pair[1] for pair in data_pairs]
        unique_values = list(set(values))
        
        if len(unique_values) == 1:
            # All columns have the same value - create concise statement
            common_value = unique_values[0]
            if len(data_pairs) == 1:
                return f"{primary_item}: {common_value}."
            else:
                # Multiple columns with same value
                column_names = [pair[0] for pair in data_pairs]
                if len(column_names) <= 3:
                    columns_text = ', '.join(column_names)
                    return f"{primary_item} ({columns_text}): {common_value}."
                else:
                    return f"{primary_item}: {common_value} across all categories."
        else:
            # Different values - create detailed statement
            details = []
            for column_name, value in data_pairs:
                if column_name and value:
                    details.append(f"{column_name}: {value}")
            
            if len(details) <= 3:
                return f"{primary_item} - " + "; ".join(details) + "."
            else:
                # Too many details, summarize
                return f"{primary_item} has varying specifications: " + "; ".join(details[:3]) + f" and {len(details)-3} more."

    def load_all_documents(self, directory: str = None) -> List[Document]:
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
