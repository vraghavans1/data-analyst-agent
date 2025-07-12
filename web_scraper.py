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
            
            # For highest-grossing films, we typically want the first main table
            df = tables[0]
            
            # Clean column names
            if df.columns.nlevels > 1:
                # Handle multi-level columns
                df.columns = [' '.join(col).strip() for col in df.columns.values]
            
            # Clean the column names
            df.columns = [self._clean_column_name(col) for col in df.columns]
            
            # Special handling for Wikipedia highest-grossing films table
            if 'highest-grossing' in url or 'highest_grossing' in url:
                df = self._process_wikipedia_films_table(df)
            
            # General cleaning
            df = self._clean_dataframe(df)
            
            self.logger.info(f"Successfully scraped Wikipedia table with shape: {df.shape}")
            self.logger.info(f"Columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error scraping Wikipedia table from {url}: {str(e)}")
            # Fallback to BeautifulSoup method
            return self.scrape_table_data(url, 'table.wikitable')
    
    def _process_wikipedia_films_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Special processing for Wikipedia highest-grossing films table."""
        # Common column name mappings for the films table
        column_mappings = {
            'rank': 'Rank',
            'peak': 'Peak',
            'title': 'Title',
            'film': 'Film',
            'worldwide gross': 'Gross',
            'highest-grossing film': 'Film',
            'worldwide gross (2019)': 'Gross',
            'year': 'Year',
            'reference(s)': 'References'
        }
        
        # Rename columns based on common patterns
        new_columns = {}
        for col in df.columns:
            col_lower = col.lower()
            for pattern, new_name in column_mappings.items():
                if pattern in col_lower:
                    new_columns[col] = new_name
                    break
        
        if new_columns:
            df = df.rename(columns=new_columns)
        
        # Ensure we have the critical columns
        if 'Rank' not in df.columns:
            # Try to find a column that looks like rank
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64'] and df[col].min() == 1:
                    df['Rank'] = df[col]
                    break
            else:
                # Create rank column if not found
                df['Rank'] = range(1, len(df) + 1)
        
        if 'Peak' not in df.columns:
            # Peak might be the same as Rank for some tables
            df['Peak'] = df['Rank']
        
        return df
    
    def _clean_column_name(self, col: str) -> str:
        """Clean a column name."""
        # Remove extra whitespace
        col = ' '.join(col.split())
        # Remove footnote markers
        col = re.sub(r'\[\d+\]', '', col)
        # Remove special characters at the end
        col = col.rstrip('*†‡§¶')
        return col.strip()
    
    def _table_to_dataframe(self, table) -> pd.DataFrame:
        """Convert BeautifulSoup table to pandas DataFrame."""
        rows = []
        headers = []
        
        # Extract headers
        header_row = table.find('tr')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            headers = [self._clean_column_name(h) for h in headers]
        
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
        
        # Special handling for currency columns
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['gross', 'revenue', 'box office', 'earnings']):
                df[col] = self._clean_currency_column(df[col])
        
        # Special handling for year columns
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['year', 'date', 'release']):
                df[col] = self._extract_year_from_column(df[col])
        
        # Convert numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                # Only convert if at least 50% of values are numeric
                if numeric_col.notna().sum() > len(df) * 0.5:
                    df[col] = numeric_col
        
        return df
    
    def _clean_currency_column(self, series: pd.Series) -> pd.Series:
        """Clean currency columns to numeric values."""
        # Create a copy to avoid modifying the original
        cleaned = series.copy()
        
        # Convert to string first
        cleaned = cleaned.astype(str)
        
        # Remove currency symbols
        cleaned = cleaned.str.replace('$', '', regex=False)
        cleaned = cleaned.str.replace('£', '', regex=False)
        cleaned = cleaned.str.replace('€', '', regex=False)
        cleaned = cleaned.str.replace('¥', '', regex=False)
        
        # Handle billion/million notation
        # Convert "2.5 billion" to 2500000000
        def convert_value(val):
            val = str(val).lower().strip()
            if 'billion' in val:
                num = re.search(r'([\d.]+)\s*billion', val)
                if num:
                    return float(num.group(1)) * 1_000_000_000
            elif 'million' in val:
                num = re.search(r'([\d.]+)\s*million', val)
                if num:
                    return float(num.group(1)) * 1_000_000
            else:
                # Remove commas and try to convert
                val = val.replace(',', '')
                try:
                    return float(val)
                except:
                    return None
            return None
        
        # Apply conversion
        cleaned = cleaned.apply(convert_value)
        
        return cleaned
    
    def _extract_year_from_column(self, series: pd.Series) -> pd.Series:
        """Extract year from a column that might contain dates or mixed text."""
        def extract_year(val):
            val = str(val)
            # Look for 4-digit year
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', val)
            if year_match:
                return int(year_match.group(1))
            return None
        
        return series.apply(extract_year)
    
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