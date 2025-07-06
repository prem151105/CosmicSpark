import os
import time
import requests
from typing import List, Dict
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

# Use relative imports for local modules
from .pdf_processor import PDFProcessor
from .content_extractor import ContentExtractor

# Import logger from the project root
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import get_logger, setup_logger

# Initialize logger
setup_logger()
logger = get_logger(__name__)

class MOSDACCrawler:
    def __init__(self, base_url="https://www.mosdac.gov.in"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MOSDAC-HelpBot-Crawler/1.0'
        })
        self.visited_urls = set()
        self.allowed_domain = urlparse(base_url).netloc
        self.output_dir = "data/raw"
        self.pdf_processor = PDFProcessor()
        self.content_extractor = ContentExtractor()

    def crawl_all_pages(self) -> List[str]:
        """Crawl all pages, PDFs, and documents"""
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            start_urls = [self.base_url]
            collected_urls = []
            
            for url in start_urls:
                if url not in self.visited_urls:
                    try:
                        content, metadata = self._fetch_page(url)
                        self.visited_urls.add(url)
                        collected_urls.append(url)
                        
                        # Save HTML content and extract structured data
                        filename = self._url_to_filename(url)
                        html_path = os.path.join(self.output_dir, filename)
                        self.save_content(content, html_path)
                        self.content_extractor.extract_from_html(html_path, source_url=url)
                        
                        # Extract links for further crawling
                        new_urls = self._extract_links(content, url)
                        start_urls.extend([u for u in new_urls if u not in self.visited_urls])
                        
                        # Handle PDFs
                        pdf_urls = [u for u in new_urls if u.endswith('.pdf')]
                        for pdf_url in pdf_urls:
                            self._process_pdf(pdf_url)
                            
                    except Exception as e:
                        logger.error(f"Error crawling {url}: {str(e)}")
                    
                    # Respectful crawling delay
                    time.sleep(1)
                    
            return collected_urls
        
        except Exception as e:
            logger.error(f"Crawl failed: {str(e)}")
            return []

    def _fetch_page(self, url: str) -> tuple:
        """Fetch page content and metadata"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            metadata = {
                'title': soup.title.string if soup.title else '',
                'description': self._get_meta_description(soup),
                'url': url,
                'timestamp': time.time()
            }
            
            return soup, metadata
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {str(e)}")
            return None, {}

    def _get_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description from page"""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        return meta_desc['content'] if meta_desc and meta_desc.get('content') else ''

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all valid links from page"""
        links = []
        if not soup:
            return links
            
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)
            
            # Filter for same domain and valid URLs
            if (urlparse(full_url).netloc == self.allowed_domain and
                full_url not in self.visited_urls and
                not any(ext in full_url for ext in ['#', 'javascript:', 'mailto:'])):
                links.append(full_url)
                
        return links

    def _process_pdf(self, pdf_url: str) -> None:
        """Download PDF and delegate processing to PDFProcessor"""
        try:
            response = self.session.get(pdf_url, timeout=10)
            response.raise_for_status()
            
            filename = self._url_to_filename(pdf_url)
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Process PDF using PDFProcessor
            self.pdf_processor.process_pdf(filepath, source_url=pdf_url)
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_url}: {str(e)}")

    def extract_metadata(self, url: str) -> Dict:
        """Extract page metadata, title, description"""
        _, metadata = self._fetch_page(url)
        return metadata

    def save_content(self, content: BeautifulSoup, filepath: str) -> None:
        """Save raw HTML content to file"""
        try:
            if content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(str(content))
                logger.info(f"Saved raw HTML to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save content to {filepath}: {str(e)}")

    def _url_to_filename(self, url: str) -> str:
        """Convert URL to safe filename"""
        parsed = urlparse(url)
        path = parsed.path.replace('/', '_').strip('_')
        return f"{parsed.netloc}_{path or 'index'}.html"