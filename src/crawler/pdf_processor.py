import fitz
import os
from typing import Dict, Optional
from utils.logger import get_logger
import json
from urllib.parse import urlparse

logger = get_logger(__name__)

class PDFProcessor:
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_pdf(self, pdf_path: str, source_url: Optional[str] = None) -> Dict:
        """Process a single PDF file, extract text and metadata, and save to output_dir."""
        try:
            # Open PDF
            doc = PyMuPDF.open(pdf_path)
            text = ""
            metadata = {
                "source_url": source_url or "",
                "filename": os.path.basename(pdf_path),
                "page_count": doc.page_count,
                "pdf_metadata": doc.metadata
            }

            # Extract text from each page
            for page in doc:
                text += page.get_text("text") + "\n"

            doc.close()

            # Save extracted text
            output_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".txt"
            output_path = os.path.join(self.output_dir, output_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info(f"Saved extracted text to {output_path}")

            # Save metadata as JSON
            metadata_filename = os.path.splitext(os.path.basename(pdf_path))[0] + "_metadata.json"
            metadata_path = os.path.join(self.output_dir, metadata_filename)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")

            return {
                "text_path": output_path,
                "metadata_path": metadata_path,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {str(e)}")
            return {
                "text_path": "",
                "metadata_path": "",
                "status": "error",
                "error": str(e)
            }

    def batch_process_pdfs(self, pdf_files: list) -> list:
        """Process multiple PDF files in batch."""
        results = []
        for pdf_path in pdf_files:
            result = self.process_pdf(pdf_path)
            results.append(result)
        return results