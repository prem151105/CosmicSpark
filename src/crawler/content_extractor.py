from bs4 import BeautifulSoup
import os
import json
from typing import Dict, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class ContentExtractor:
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_from_html(self, html_path: str, source_url: Optional[str] = None) -> Dict:
        """Extract structured content from an HTML file."""
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                content = f.read()
            soup = BeautifulSoup(content, "html.parser")

            # Extract metadata
            metadata = {
                "source_url": source_url or "",
                "filename": os.path.basename(html_path),
                "title": soup.title.string if soup.title else "",
                "description": self._get_meta_description(soup)
            }

            # Extract structured content
            structured_content = {
                "headings": self._extract_headings(soup),
                "paragraphs": self._extract_paragraphs(soup),
                "links": self._extract_links(soup)
            }

            # Save extracted text
            output_filename = os.path.splitext(os.path.basename(html_path))[0] + "_content.txt"
            output_path = os.path.join(self.output_dir, output_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(soup.get_text(strip=True))
            logger.info(f"Saved extracted text to {output_path}")

            # Save structured content and metadata as JSON
            output_json = {
                "metadata": metadata,
                "content": structured_content
            }
            json_filename = os.path.splitext(os.path.basename(html_path))[0] + "_structured.json"
            json_path = os.path.join(self.output_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output_json, f, indent=2)
            logger.info(f"Saved structured content to {json_path}")

            return {
                "text_path": output_path,
                "json_path": json_path,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Failed to extract content from {html_path}: {str(e)}")
            return {
                "text_path": "",
                "json_path": "",
                "status": "error",
                "error": str(e)
            }

    def _get_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description from HTML."""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        return meta_desc["content"] if meta_desc and meta_desc.get("content") else ""

    def _extract_headings(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract all headings (h1-h6) from HTML."""
        headings = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            headings.append({
                "level": tag.name,
                "text": tag.get_text(strip=True)
            })
        return headings

    def _extract_paragraphs(self, soup: BeautifulSoup) -> List[str]:
        """Extract all paragraph texts from HTML."""
        return [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]

    def _extract_links(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract all links from HTML."""
        return [{"href": a.get("href", ""), "text": a.get_text(strip=True)} for a in soup.find_all("a", href=True)]

    def batch_extract_html(self, html_files: List[str]) -> List[Dict]:
        """Process multiple HTML files in batch."""
        results = []
        for html_path in html_files:
            result = self.extract_from_html(html_path)
            results.append(result)
        return results