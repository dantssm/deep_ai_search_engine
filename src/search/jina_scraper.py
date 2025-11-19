import httpx
import asyncio
from typing import List, Dict, Optional
from src.utils.logger import log_scrape

class JinaWebScraper:
    """Web scraper using Jina AI's reader service."""

    BASE_URL = "https://r.jina.ai/"
    TIMEOUT = 10.0
    
    def __init__(self, max_content_length: int = 6000):
        self.max_content_length = max_content_length
    
    async def scrape_url(self, url: str) -> Optional[str]:
        """
        Scrapes the content of a single URL using Jina's web scraper.

        Args:
            url: The URL to scrape.

        Returns:
            Optional[str]: The scraped content as a string, or None if scraping failed.
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
                    return None
                log_scrape(f"HTTP {e.response.status_code} for {url[:50]}...", level="warning")
                return None
            
            except Exception as e:
                return None
    
    async def scrape_multiple(self, urls: List[str], max_concurrent: int = 10) -> List[Dict]:
        """
        Scrapes multiple URLs concurrently.

        Args:
            urls: A list of URLs.
            max_concurrent: Maximum number of concurrent requests.

        Returns:
            List[Dict]: A list of dictionaries with 'url' and 'content' keys of scraped data:
                [{"url": str, "content": str}, ...]
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_one(url: str):
            async with semaphore:
                content = await self.scrape_url(url)
                if content:
                    return {"url": url, "content": content}
                return None
        
        log_scrape(f"Scraping {len(urls)} URLs (max {max_concurrent} at once)...")
        
        tasks = [scrape_one(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        valid_results = [r for r in results if r is not None]
        
        log_scrape(f"Successfully scraped {len(valid_results)}/{len(urls)} URLs")
        return valid_results