import unittest
from unittest.mock import patch, Mock, mock_open
import sys
from pathlib import Path

# Add src directory to sys.path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.crawler.web_crawler import MOSDACCrawler
from src.crawler.pdf_processor import PDFProcessor
from src.crawler.content_extractor import ContentExtractor
from src.logger import get_logger, setup_logger

setup_logger()
logger = get_logger(__name__)

class TestMOSDACCrawler(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.base_url = "https://www.mosdac.gov.in"
        self.crawler = MOSDACCrawler(base_url=self.base_url)
        self.sample_html = """
        <html>
            <head>
                <title>Test Page</title>
                <meta name="description" content="Test Description">
            </head>
            <body>
                <h1>Test Heading</h1>
                <p>Test Paragraph</p>
                <a href="/page1">Page 1</a>
                <a href="document.pdf">PDF Link</a>
                <a href="javascript:void(0)">Invalid Link</a>
            </body>
        </html>
        """
        self.sample_metadata = {
            "title": "Test Page",
            "description": "Test Description",
            "url": self.base_url,
            "timestamp": 1625313600.0
        }

    @patch("requests.Session.get")
    def test_fetch_page(self, mock_get):
        """Test _fetch_page method."""
        mock_response = Mock()
        mock_response.text = self.sample_html
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch("time.time", return_value=1625313600.0):
            soup, metadata = self.crawler._fetch_page(self.base_url)

        self.assertIsInstance(soup, BeautifulSoup)
        self.assertEqual(soup.title.string, "Test Page")
        self.assertEqual(metadata["title"], "Test Page")
        self.assertEqual(metadata["description"], "Test Description")
        self.assertEqual(metadata["url"], self.base_url)
        logger.info("test_fetch_page passed")

    @patch("requests.Session.get")
    def test_fetch_page_error(self, mock_get):
        """Test _fetch_page with request error."""
        mock_get.side_effect = requests.RequestException("Connection error")

        soup, metadata = self.crawler._fetch_page(self.base_url)
        self.assertIsNone(soup)
        self.assertEqual(metadata, {})
        logger.info("test_fetch_page_error passed")

    def test_extract_links(self):
        """Test _extract_links method."""
        soup = BeautifulSoup(self.sample_html, "html.parser")
        links = self.crawler._extract_links(soup, self.base_url)
        expected_links = [
            "https://www.mosdac.gov.in/page1",
            "https://www.mosdac.gov.in/document.pdf"
        ]
        self.assertEqual(links, expected_links)
        logger.info("test_extract_links passed")

    @patch("requests.Session.get")
    @patch.object(PDFProcessor, "process_pdf")
    @patch("builtins.open", new_callable=mock_open)
    def test_process_pdf(self, mock_file, mock_process_pdf, mock_get):
        """Test _process_pdf method."""
        pdf_url = "https://www.mosdac.gov.in/document.pdf"
        mock_response = Mock()
        mock_response.content = b"PDF content"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        mock_process_pdf.return_value = {
            "text_path": "data/processed/document.txt",
            "metadata_path": "data/processed/document_metadata.json",
            "status": "success"
        }

        self.crawler._process_pdf(pdf_url)
        mock_file.assert_called_with("data/raw/www.mosdac.gov.in_document.pdf", "wb")
        mock_process_pdf.assert_called_with("data/raw/www.mosdac.gov.in_document.pdf", source_url=pdf_url)
        logger.info("test_process_pdf passed")

    @patch("builtins.open", new_callable=mock_open)
    def test_save_content(self, mock_file):
        """Test save_content method."""
        soup = BeautifulSoup(self.sample_html, "html.parser")
        filepath = "data/raw/www.mosdac.gov.in_index.html"
        self.crawler.save_content(soup, filepath)
        mock_file.assert_called_with(filepath, "w", encoding="utf-8")
        mock_file().write.assert_called_with(str(soup))
        logger.info("test_save_content passed")

    @patch("requests.Session.get")
    @patch.object(PDFProcessor, "process_pdf")
    @patch.object(ContentExtractor, "extract_from_html")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_crawl_all_pages(self, mock_makedirs, mock_file, mock_extract_html, mock_process_pdf, mock_get):
        """Test crawl_all_pages method."""
        mock_response = Mock()
        mock_response.text = self.sample_html
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        mock_process_pdf.return_value = {"status": "success"}
        mock_extract_html.return_value = {"status": "success"}

        with patch("time.time", return_value=1625313600.0):
            urls = self.crawler.crawl_all_pages()

        self.assertIn(self.base_url, urls)
        self.assertTrue(mock_extract_html.called)
        self.assertTrue(mock_process_pdf.called)
        logger.info("test_crawl_all_pages passed")

    def test_url_to_filename(self):
        """Test _url_to_filename method."""
        url = "https://www.mosdac.gov.in/about/team"
        filename = self.crawler._url_to_filename(url)
        self.assertEqual(filename, "www.mosdac.gov.in_about_team.html")
        logger.info("test_url_to_filename passed")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

if __name__ == "__main__":
    unittest.main()