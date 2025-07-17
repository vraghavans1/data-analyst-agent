import requests
import pandas as pd
from bs4 import BeautifulSoup
import trafilatura
import logging
import re
from typing import Optional, List, Dict, Any
from io import StringIO

class WebScraper:
    """Handles web scraping operations for data sourcing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_website_text_content(self, url: str) -> str:
        """
        Extract main text content from a website using trafilatura.
        The text content is extracted using trafilatura and easier to understand.
        """
        try:
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded)
            return text or ""
        except Exception as e:
            self.logger.error(f"Error extracting text from {url}: {str(e)}")
            return ""
    
    def scrape_table_data(self, url: str, table_selector: str = None) -> Optional[pd.DataFrame]:
        """Scrape table data from a webpage."""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find tables
            if table_selector:
                tables = soup.select(table_selector)
            else:
                tables = soup.find_all('table')
            
            if not tables:
                self.logger.warning(f"No tables found on {url}")
                return None
            
            # Convert the first table to DataFrame
            table = tables[0]
            df = self._table_to_dataframe(table)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error scraping table from {url}: {str(e)}")
            return None
    
    def scrape_wikipedia_table(self, url: str) -> Optional[pd.DataFrame]:
        """Specifically scrape Wikipedia tables with better parsing."""
        try:
            # Use pandas read_html for Wikipedia tables - it's more robust
            tables = pd.read_html(url)
            
            if not tables:
                return None
            
            # Usually the first table is the main data table
            df = tables[0]
            
            # Clean column names
            if df.columns.nlevels > 1:
                # Handle multi-level columns
                df.columns = [' '.join(col).strip() for col in df.columns.values]
            
            # Clean data
            df = self._clean_dataframe(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error scraping Wikipedia table from {url}: {str(e)}")
            # Fallback to BeautifulSoup method
            return self.scrape_table_data(url, 'table.wikitable')
    
    def _table_to_dataframe(self, table) -> pd.DataFrame:
        """Convert BeautifulSoup table to pandas DataFrame."""
        rows = []
        headers = []
        
        # Extract headers
        header_row = table.find('tr')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        
        # Extract data rows
        for row in table.find_all('tr')[1:]:  # Skip header row
            cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
            if cells:
                rows.append(cells)
        
        # Create DataFrame
        if rows and headers:
            # Ensure all rows have the same length as headers
            max_cols = len(headers)
            for i, row in enumerate(rows):
                if len(row) < max_cols:
                    rows[i].extend([''] * (max_cols - len(row)))
                elif len(row) > max_cols:
                    rows[i] = row[:max_cols]
            
            df = pd.DataFrame(rows, columns=headers)
            return self._clean_dataframe(df)
        
        return pd.DataFrame()
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize DataFrame data."""
        # Clean column names
        df.columns = [col.strip().replace('\n', ' ').replace('\r', ' ') for col in df.columns]
        
        # Clean data values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                df[col] = df[col].str.replace(r'\[.*?\]', '', regex=True)  # Remove citation marks
        
        # Convert numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric - handle more complex cases
                numeric_col = df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '')
                numeric_col = numeric_col.str.replace(r'[^\d.-]', '', regex=True)  # Remove non-numeric characters
                numeric_col = pd.to_numeric(numeric_col, errors='coerce')  # Use coerce to handle invalid values
                if numeric_col.notna().any():  # If any values were successfully converted
                    df[col] = numeric_col
        
        # Convert date columns
        for col in df.columns:
            if 'date' in col.lower() or 'year' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='ignore')
        
        return df
    
    def scrape_multiple_tables(self, url: str) -> List[pd.DataFrame]:
        """Scrape multiple tables from a webpage."""
        try:
            tables = pd.read_html(url)
            return [self._clean_dataframe(table) for table in tables]
        except Exception as e:
            self.logger.error(f"Error scraping multiple tables from {url}: {str(e)}")
            return []
    
    def scrape_list_data(self, url: str, list_selector: str = None) -> List[Dict[str, Any]]:
        """Scrape list data from a webpage."""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find lists
            if list_selector:
                lists = soup.select(list_selector)
            else:
                lists = soup.find_all(['ul', 'ol'])
            
            data = []
            for list_elem in lists:
                items = list_elem.find_all('li')
                for item in items:
                    text = item.get_text(strip=True)
                    if text:
                        data.append({'text': text})
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error scraping list from {url}: {str(e)}")
            return []
