"""
Lightweight web scraper for Vercel deployment
Uses only BeautifulSoup without trafilatura dependency
"""

import requests
from bs4 import BeautifulSoup
import json
import logging
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    """
    Lightweight web scraper using BeautifulSoup.
    Optimized for Vercel deployment without heavy dependencies.
    """
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_data(self, url: str) -> Dict[str, Any]:
        """
        Scrape data from a URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dict containing scraped data
        """
        try:
            logger.info(f"Scraping data from: {url}")
            
            # Fetch the page
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract data
            data = {
                'url': url,
                'title': self._extract_title(soup),
                'content': self._extract_content(soup),
                'tables': self._extract_tables(soup),
                'metadata': self._extract_metadata(soup)
            }
            
            logger.info(f"Successfully scraped data from {url}")
            return data
            
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return {'error': f"Failed to fetch data from {url}: {str(e)}"}
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return {'error': f"Failed to scrape data from {url}: {str(e)}"}
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Try h1 as fallback
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return "No title found"
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from the page."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'main',
            'article',
            '.content',
            '.main-content',
            '#content',
            '#main'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                return content_elem.get_text().strip()
        
        # Fallback to body
        body = soup.find('body')
        if body:
            return body.get_text().strip()
        
        return soup.get_text().strip()
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract tables from the page."""
        tables = []
        
        for i, table in enumerate(soup.find_all('table')):
            table_data = []
            
            # Get table headers
            headers = []
            header_row = table.find('tr')
            if header_row:
                for th in header_row.find_all(['th', 'td']):
                    headers.append(th.get_text().strip())
            
            # Get table rows
            for row in table.find_all('tr'):
                row_data = []
                for cell in row.find_all(['td', 'th']):
                    row_data.append(cell.get_text().strip())
                
                if row_data:
                    table_data.append(row_data)
            
            if table_data:
                tables.append({
                    'name': f'Table {i+1}',
                    'headers': headers,
                    'data': table_data,
                    'row_count': len(table_data),
                    'column_count': len(headers) if headers else (len(table_data[0]) if table_data else 0)
                })
        
        return tables
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from the page."""
        metadata = {}
        
        # Meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name', meta.get('property', ''))
            content = meta.get('content', '')
            if name and content:
                metadata[name] = content
        
        # Get all headings
        headings = []
        for i in range(1, 7):  # h1 to h6
            for heading in soup.find_all(f'h{i}'):
                headings.append({
                    'level': i,
                    'text': heading.get_text().strip()
                })
        
        metadata['headings'] = headings
        
        # Count elements
        metadata['element_counts'] = {
            'paragraphs': len(soup.find_all('p')),
            'links': len(soup.find_all('a')),
            'images': len(soup.find_all('img')),
            'tables': len(soup.find_all('table')),
            'lists': len(soup.find_all(['ul', 'ol']))
        }
        
        return metadata
    
    def get_website_text_content(self, url: str) -> str:
        """
        Extract clean text content from a website.
        
        Args:
            url: URL to scrape
            
        Returns:
            Clean text content
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from {url}: {str(e)}")
            return f"Error extracting text from {url}: {str(e)}"
    
    def scrape_wikipedia_data(self, topic: str) -> Dict[str, Any]:
        """
        Scrape data from Wikipedia for a given topic.
        
        Args:
            topic: Topic to search for
            
        Returns:
            Wikipedia data
        """
        # Construct Wikipedia URL
        url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        
        try:
            logger.info(f"Scraping Wikipedia data for: {topic}")
            return self.scrape_data(url)
            
        except Exception as e:
            logger.error(f"Error scraping Wikipedia for {topic}: {str(e)}")
            return {'error': f"Failed to scrape Wikipedia for {topic}: {str(e)}"}

# Utility functions
def extract_tables_from_html(html_content: str) -> List[Dict[str, Any]]:
    """Extract tables from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    scraper = WebScraper()
    return scraper._extract_tables(soup)

def clean_text_content(html_content: str) -> str:
    """Clean text content from HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove unwanted elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        element.decompose()
    
    # Extract and clean text
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return ' '.join(chunk for chunk in chunks if chunk)