import requests
import os
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# Tentative: Will be moved to a more appropriate location later,
# possibly a central configuration module or loaded from .env
DATA_RAW_DIR = "data/raw"

class MOSDACCrawler:
    def __init__(self, base_url="https://www.mosdac.gov.in"):
        self.base_url = base_url
        self.session = requests.Session()
        # Basic headers, can be expanded
        self.session.headers.update({
            "User-Agent": "MOSDACHelpBotCrawler/1.0 (+http://example.com/botinfo)"
        })
        self.visited_urls = set()

        if not os.path.exists(DATA_RAW_DIR):
            os.makedirs(DATA_RAW_DIR)
            print(f"Created directory: {DATA_RAW_DIR}")

    def crawl_all_pages(self):
        """
        Crawl all accessible pages, PDFs, and documents starting from the base_url.
        This will be a breadth-first or depth-first crawl.
        For now, it will just be a placeholder.
        """
        print(f"Starting crawl from: {self.base_url}")
        # To be implemented:
        # - Queue for URLs to visit
        # - Logic to fetch page content
        # - Logic to find new links on the page
        # - Handling different content types (HTML, PDF, DOC)
        # - Respecting robots.txt (if available and configured)
        # - Error handling (timeouts, HTTP errors)
        # - Depth limiting / page count limiting

        # Example of fetching a single page:
        try:
            response = self.session.get(self.base_url, timeout=10)
            response.raise_for_status() # Raise an exception for HTTP errors

            if self.base_url not in self.visited_urls:
                self.visited_urls.add(self.base_url)
                print(f"Successfully fetched {self.base_url}")

                # Process and save content (simplified for now)
                content = response.text
                metadata = self.extract_metadata(self.base_url, content)

                # Create a filename based on the URL path
                parsed_url = urlparse(self.base_url)
                path_parts = [part for part in parsed_url.path.split('/') if part]
                if not path_parts or not path_parts[-1]:
                    filename = "index.html"
                else:
                    filename = "_".join(path_parts) + ".html" # Basic sanitization

                filepath = os.path.join(DATA_RAW_DIR, filename)
                self.save_content(content, filepath, metadata)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {self.base_url}: {e}")

        print("Crawling process (stub) finished.")
        # Placeholder for returning crawled data or status
        return {"status": "success", "pages_crawled": len(self.visited_urls)}


    def extract_metadata(self, url, page_content=None):
        """
        Extract page metadata such as title, description, keywords, etc.
        """
        metadata = {"url": url, "title": "", "description": "", "keywords": ""}

        if not page_content:
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                page_content = response.text
            except requests.exceptions.RequestException as e:
                print(f"Error fetching content for metadata extraction from {url}: {e}")
                return metadata

        soup = BeautifulSoup(page_content, 'html.parser')

        if soup.title and soup.title.string:
            metadata["title"] = soup.title.string.strip()

        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag and desc_tag.get("content"):
            metadata["description"] = desc_tag.get("content").strip()

        keywords_tag = soup.find("meta", attrs={"name": "keywords"})
        if keywords_tag and keywords_tag.get("content"):
            metadata["keywords"] = keywords_tag.get("content").strip()

        # Add more metadata extraction logic as needed (e.g., OpenGraph tags)

        return metadata

    def save_content(self, content, filepath, metadata=None):
        """
        Save crawled content to a structured format (e.g., HTML file, JSON for metadata).
        The content could be HTML, PDF binary data, etc.
        """
        try:
            # For HTML, save as .html
            # For PDFs, content would be binary, save as .pdf
            # For now, assuming content is text (HTML)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Saved content to {filepath}")

            # Save metadata as a separate .meta.json file or include in a structured file
            if metadata:
                meta_filepath = filepath + ".meta.json"
                import json
                with open(meta_filepath, "w", encoding="utf-8") as mf:
                    json.dump(metadata, mf, indent=4)
                print(f"Saved metadata to {meta_filepath}")

        except IOError as e:
            print(f"Error saving content to {filepath}: {e}")

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    crawler = MOSDACCrawler()

    # Test fetching and saving the base page
    # crawler.crawl_all_pages()

    # Test metadata extraction (assuming you have a page's content)
    # test_url = "https://www.mosdac.gov.in"
    # try:
    #     response = requests.get(test_url)
    #     response.raise_for_status()
    #     metadata = crawler.extract_metadata(test_url, response.text)
    #     print("\nExtracted Metadata:")
    #     import json
    #     print(json.dumps(metadata, indent=2))
    # except requests.RequestException as e:
    #     print(f"Could not fetch {test_url} for metadata test: {e}")

    print("\nMOSDACCrawler basic structure created. Further implementation needed for full functionality.")
