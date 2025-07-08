"""
PDF Processor for extracting text and metadata from PDF documents
"""

import io
import re
import fitz  # PyMuPDF
from typing import Dict, Optional, Tuple
from datetime import datetime
from loguru import logger

class PDFProcessor:
    """Process PDF documents to extract text and metadata"""
    
    def __init__(self):
        """Initialize the PDF processor"""
        pass
    
    def process_pdf(self, pdf_content: bytes, url: str = None) -> Tuple[str, Dict[str, str]]:
        """
        Process a PDF document and extract text and metadata
        
        Args:
            pdf_content: Binary content of the PDF file
            url: (Optional) URL where the PDF was downloaded from
            
        Returns:
            Tuple of (extracted_text, metadata_dict)
        """
        return self.extract_text(pdf_content, url)
        
    def extract_text(self, pdf_content: bytes, url: str = None) -> Tuple[str, Dict[str, str]]:
        """
        Extract text from PDF document (alias for process_pdf)
        
        Args:
            pdf_content: Binary content of the PDF file
            url: (Optional) URL where the PDF was downloaded from
            
        Returns:
            Tuple of (extracted_text, metadata_dict)
        """
        if not isinstance(pdf_content, bytes):
            raise ValueError("pdf_content must be bytes")
            
        if not url:
            url = "unknown_source.pdf"
        try:
            # Open the PDF from bytes
            with fitz.open(stream=pdf_content, filetype="pdf") as doc:
                # Extract text from all pages
                text_parts = []
                for page in doc:
                    text = page.get_text()
                    if text.strip():
                        text_parts.append(text.strip())
                
                # Combine all text parts
                full_text = "\n\n".join(text_parts)
                
                # Extract metadata
                metadata = self._extract_metadata(doc.metadata, url)
                
                return full_text, metadata
                
        except Exception as e:
            logger.error(f"Error processing PDF from {url}: {e}")
            return "", {}
    
    def _extract_metadata(self, pdf_metadata: Dict, url: str = None) -> Dict[str, str]:
        """
        Extract and format metadata from PDF
        
        Args:
            pdf_metadata: Raw metadata from PyMuPDF
            url: (Optional) Source URL of the PDF
            
        Returns:
            Dictionary of formatted metadata
        """
        # Format dates to ISO format
        def format_date(date_str: Optional[str]) -> str:
            if not date_str:
                return ""
            try:
                # Try to parse common PDF date formats
                date_str = date_str.replace("D:", "").strip()
                if "+" in date_str:
                    date_str = date_str.split("+")[0]
                return date_str
            except:
                return ""
        
        # Extract basic metadata
        # Generate a fallback title from URL or use a default
        fallback_title = 'untitled_document.pdf'
        if url:
            fallback_title = url.split('/')[-1] or fallback_title
            
        metadata = {
            'title': pdf_metadata.get('title', '').strip() or fallback_title,
            'author': pdf_metadata.get('author', '').strip(),
            'subject': pdf_metadata.get('subject', '').strip(),
            'keywords': pdf_metadata.get('keywords', '').strip(),
            'creation_date': format_date(pdf_metadata.get('creationDate', '')),
            'modification_date': format_date(pdf_metadata.get('modDate', '')),
            'producer': pdf_metadata.get('producer', '').strip(),
            'creator': pdf_metadata.get('creator', '').strip(),
            'url': url or 'local_file',
            'processing_date': datetime.utcnow().isoformat(),
            'format': 'PDF',
            'page_count': pdf_metadata.get('page_count', 0)
        }
        
        # Clean up metadata
        for key, value in metadata.items():
            if isinstance(value, str):
                # Remove any non-printable characters
                value = re.sub(r'[^\x20-\x7E]', ' ', value)
                metadata[key] = ' '.join(value.split())
        
        return metadata