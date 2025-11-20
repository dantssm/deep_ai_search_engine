"""Jina AI Web Scraper wrapper"""

import httpx
import asyncio
from typing import List, Dict, Optional
from src.utils.logger import log_scrape

class JinaWebScraper:
    """Client for Jina AI's Reader API"""

    BASE_URL = "https://r.jina.ai/"
    TIMEOUT = 10.0
    
    def __init__(self, max_content_length: int = 6000):
        """
        Initialize the scraper.

        Args:
            max_content_length (int): Maximum characters to keep per page.
        """
        self.max_content_length = max_content_length
    
    async def scrape_url(self, url: str) -> Optional[str]:
        """
        Scrapes a single URL.

        Args:
            url (str): The URL to scrape.

        Returns:
            Optional[str]: The text content, or None if scraping failed.
        """
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
            try:
                jina_url = f"{self.BASE_URL}{url}"
                
                response = await client.get(jina_url, follow_redirects=True)
                response.raise_for_status()
                
                content = response.text
                
                if len(content) > self.max_content_length:
                    content = content[:self.max_content_length] + "..."
                
                log_scrape(f"Scraped {len(content)} chars from {url[:50]}...")
                return content
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    log_scrape(f"Rate limited (429) for {url[:50]}...", level="warning")
                    return None
                log_scrape(f"HTTP {e.response.status_code} for {url[:50]}...", level="warning")
                return None
            
            except Exception:
                return None
    
    async def scrape_multiple(self, urls: List[str], max_concurrent: int = 10) -> List[Dict]:
        """
        Scrapes multiple URLs concurrently with a semaphore limit.

        Args:
            urls (List[str]): List of URLs to scrape.
            max_concurrent (int): Max number of simultaneous requests.

        Returns:
            List[Dict]: A list of successful scrapes. Each dict contains:
                - url (str): The source URL.
                - content (str): The scraped text.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _scrape_with_semaphore(target_url: str) -> Optional[Dict]:
            async with semaphore:
                content = await self.scrape_url(target_url)
                if content:
                    return {"url": target_url, "content": content}
                return None
        
        log_scrape(f"Scraping {len(urls)} URLs")
        
        tasks = [_scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        valid_results = [r for r in results if r is not None]
        
        log_scrape(f"Successfully scraped {len(valid_results)}/{len(urls)} URLs")
        return valid_results