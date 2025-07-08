"""
Advanced Web Crawler for MOSDAC Scientific Content
Implements intelligent crawling with content extraction and processing
"""

import asyncio
import aiohttp
import aiofiles
import random
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from urllib.parse import urljoin, urlparse, parse_qs
from pathlib import Path
import re
import json
import hashlib
from bs4 import BeautifulSoup
import requests
from loguru import logger

@dataclass
class CrawledDocument:
    """Structured representation of crawled document"""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    content_type: str  # html, pdf, doc, image
    file_size: int
    crawl_timestamp: datetime
    content_hash: str
    links: List[str]
    images: List[str]

@dataclass
class CrawlConfig:
    """Configuration for web crawling"""
    max_pages: int = 1000
    max_depth: int = 3
    delay_range: Tuple[float, float] = (1.0, 3.0)
    respect_robots_txt: bool = True
    user_agent: str = "MOSDAC-Crawler/1.0"
    timeout: int = 30
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_domains: List[str] = None
    excluded_patterns: List[str] = None

    def __post_init__(self):
        if self.allowed_domains is None:
            self.allowed_domains = ['mosdac.gov.in', 'www.mosdac.gov.in']
        if self.excluded_patterns is None:
            self.excluded_patterns = [
                r'/login', r'/admin', r'\.css$', r'\.js$',
                r'/images/', r'/css/', r'/js/'
            ]

class AdvancedWebCrawler:
    """Advanced web crawler with intelligent content extraction"""
    
    def __init__(self, config: CrawlConfig = None):
        self.config = config or CrawlConfig()
        self.visited_urls: Set[str] = set()
        self.crawled_documents: List[CrawledDocument] = []
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info("Advanced Web Crawler initialized")
    
    async def crawl_website(self, start_urls: List[str]) -> List[CrawledDocument]:
        """Main crawling method"""
        logger.info(f"Starting crawl with {len(start_urls)} seed URLs")
        
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": self.config.user_agent}
        ) as session:
            self.session = session
            
            # Initialize crawl queue
            crawl_queue = [(url, 0) for url in start_urls]  # (url, depth)
            
            while crawl_queue and len(self.crawled_documents) < self.config.max_pages:
                current_url, depth = crawl_queue.pop(0)
                
                if (current_url in self.visited_urls or 
                    depth > self.config.max_depth or
                    not self._should_crawl_url(current_url)):
                    continue
                
                try:
                    # Crawl the page
                    document = await self._crawl_page(current_url, depth)
                    
                    if document:
                        self.crawled_documents.append(document)
                        self.visited_urls.add(current_url)
                        
                        # Add discovered links to queue
                        for link in document.links:
                            if link not in self.visited_urls and len(crawl_queue) < self.config.max_pages * 2:
                                crawl_queue.append((link, depth + 1))
                        
                        logger.info(f"Crawled: {current_url} (depth: {depth}, total: {len(self.crawled_documents)}/{self.config.max_pages})")
                    
                    # Respectful delay
                    await asyncio.sleep(random.uniform(*self.config.delay_range))
                    
                except Exception as e:
                    logger.error(f"Error crawling {current_url}: {e}")
                    continue
        
        logger.info(f"Crawling completed. Total documents: {len(self.crawled_documents)}")
        return self.crawled_documents
    
    async def _crawl_page(self, url: str, depth: int) -> Optional[CrawledDocument]:
        """Crawl a single page"""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
                
                content_type = response.headers.get('content-type', '').lower()
                
                if 'text/html' in content_type:
                    return await self._crawl_html_page(url, await response.text())
                elif 'application/pdf' in content_type:
                    content = await response.read()
                    return await self._crawl_document(url, content, 'pdf')
                else:
                    logger.debug(f"Skipping unsupported content type: {content_type} for {url}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error in _crawl_page for {url}: {e}")
            return None
    
    async def _crawl_html_page(self, url: str, content: str) -> Optional[CrawledDocument]:
        """Crawl HTML page with content extraction"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup, url)
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            
            # Extract links
            links = self._extract_links(soup, url)
            
            # Extract images
            images = self._extract_images(soup, url)
            
            # Create document
            document = CrawledDocument(
                url=url,
                title=title,
                content=main_content,
                metadata=metadata,
                content_type="html",
                file_size=len(content.encode('utf-8')),
                crawl_timestamp=datetime.now(),
                content_hash=hashlib.md5(main_content.encode()).hexdigest(),
                links=links,
                images=images
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error in _crawl_html_page for {url}: {e}")
            return None
    
    async def _crawl_document(self, url: str, content: bytes, doc_type: str) -> Optional[CrawledDocument]:
        """Crawl non-HTML documents (PDF, DOC, etc.)"""
        try:
            # For PDFs, use the PDF processor
            if doc_type == 'pdf':
                from .pdf_processor import PDFProcessor
                processor = PDFProcessor()
                text_content, metadata = processor.extract_text(content, url)
            else:
                text_content = ""
                metadata = {}
            
            # Create document
            document = CrawledDocument(
                url=url,
                title=self._extract_title_from_url(url),
                content=text_content,
                metadata={
                    **metadata,
                    'file_type': doc_type,
                    'original_size': len(content)
                },
                content_type=doc_type,
                file_size=len(content),
                crawl_timestamp=datetime.now(),
                content_hash=hashlib.md5(content).hexdigest(),
                links=[],
                images=[]
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error in _crawl_document for {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)[:200]  # Limit title length
        return self._extract_title_from_url(url)
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract title from URL"""
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p]
        if path_parts:
            return path_parts[-1].replace('-', ' ').replace('_', ' ').title()
        return parsed.netloc
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Try to find main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('body') or soup
        
        # Extract text with some structure preservation
        text_content = []
        for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = element.get_text(strip=True)
            if text and len(text) > 20:  # Filter out very short text
                text_content.append(text)
        
        return '\n\n'.join(text_content)
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML"""
        return {
            'url': url,
            'domain': urlparse(url).netloc,
            'crawl_timestamp': datetime.now().isoformat(),
            'title': self._extract_title(soup, url)
        }
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract and normalize links"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            
            # Filter out non-HTTP links
            if absolute_url.startswith(('http://', 'https://')) and self._should_crawl_url(absolute_url):
                links.append(absolute_url)
        
        return list(set(links))  # Remove duplicates
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract image URLs"""
        images = []
        for img in soup.find_all('img', src=True):
            src = img['src']
            absolute_url = urljoin(base_url, src)
            
            if absolute_url.startswith(('http://', 'https://')):
                images.append(absolute_url)
        
        return list(set(images))
    
    def _should_crawl_url(self, url: str) -> bool:
        """Check if URL should be crawled"""
        parsed = urlparse(url)
        
        # Check allowed domains
        if not any(domain in parsed.netloc for domain in self.config.allowed_domains):
            return False
        
        # Check excluded patterns
        if any(re.search(pattern, url, re.IGNORECASE) for pattern in self.config.excluded_patterns):
            return False
        
        # Skip common non-content URLs
        skip_patterns = [
            r'\.(css|js|json|xml|jpg|jpeg|png|gif|ico|svg|woff|woff2|ttf|eot|pdf|docx?|xlsx?|pptx?|zip|rar|7z|gz|tar)$',
            r'/(api|ajax|login|register|admin|wp-json|wp-admin|wp-includes|wp-content|feed|rss|sitemap|robots\.txt)'
        ]
        
        if any(re.search(pattern, url, re.IGNORECASE) for pattern in skip_patterns):
            return False
        
        return True
    
    def save_crawl_results(self, output_path: str) -> None:
        """Save crawl results to JSON file"""
        results = {
            'crawl_metadata': {
                'total_documents': len(self.crawled_documents),
                'crawl_timestamp': datetime.now().isoformat(),
                'config': asdict(self.config)
            },
            'documents': [asdict(doc) for doc in self.crawled_documents]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Crawl results saved to {output_path}")
    
    def get_crawl_statistics(self) -> Dict[str, Any]:
        """Get crawling statistics"""
        if not self.crawled_documents:
            return {'total_documents': 0}
        
        content_types = {}
        total_size = 0
        
        for doc in self.crawled_documents:
            content_types[doc.content_type] = content_types.get(doc.content_type, 0) + 1
            total_size += doc.file_size
        
        return {
            'total_documents': len(self.crawled_documents),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'content_type_distribution': content_types,
            'unique_domains': len(set(urlparse(doc.url).netloc for doc in self.crawled_documents)),
            'average_content_length': sum(len(doc.content) for doc in self.crawled_documents) // max(1, len(self.crawled_documents))
        }
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'session') and self.session:
            asyncio.create_task(self.session.close())
        logger.info("Web crawler closed")

class MOSDACCrawler(AdvancedWebCrawler):
    """Specialized crawler for MOSDAC website"""
    
    def __init__(self):
        config = CrawlConfig(
            max_pages=2000,
            max_depth=4,
            allowed_domains=['mosdac.gov.in', 'www.mosdac.gov.in'],
            excluded_patterns=[
                r'/login', r'/admin', r'\.css$', r'\.js$',
                r'/images/', r'/css/', r'/js/'
            ]
        )
        super().__init__(config)

# Export main classes
__all__ = [
    'AdvancedWebCrawler',
    'MOSDACCrawler',
    'CrawledDocument',
    'CrawlConfig'
]