"""
Session-scoped In-Memory Cache. Stores search results and scraped web content to avoid redundant API calls.
"""

from typing import Dict, List, Optional
from src.search.google_search import GoogleSearcher
from src.search.jina_scraper import JinaWebScraper
from src.utils.logger import log_rag


class SimpleCache:
    
    def __init__(self):
        self.search_cache: Dict[str, List[Dict]] = {}
        self.scrape_cache: Dict[str, str] = {}
        log_rag("Cache initialized (session-based)")
    
    def get_search(self, query: str) -> Optional[List[Dict]]:
        """Retrieve cached search results for a query"""
        result = self.search_cache.get(query)
        if result:
            log_rag(f"Cache HIT for search: {query[:50]}...")
        return result
    
    def save_search(self, query: str, results: List[Dict]):
        """Store search results"""
        self.search_cache[query] = results
    
    def get_scrape(self, url: str) -> Optional[str]:
        """Retrieve cached content for a URL"""
        return self.scrape_cache.get(url)
    
    def save_scrape(self, url: str, content: str):
        """Store scraped content"""
        self.scrape_cache[url] = content
    
    def clear(self):
        """Delete all cached data"""
        self.search_cache.clear()
        self.scrape_cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Return cache usage statistics."""
        return {
            "search_entries": len(self.search_cache),
            "scrape_entries": len(self.scrape_cache)
        }


class CachedGoogleSearcher:
    
    def __init__(self, api_key: str, cse_id: str, cache: SimpleCache):
        self.searcher = GoogleSearcher(api_key, cse_id)
        self.cache = cache
    
    async def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search with cache lookup"""
        cached = self.cache.get_search(query)
        if cached:
            return cached[:num_results]
        
        results = await self.searcher.search(query, num_results)

        if results:
            self.cache.save_search(query, results)
        
        return results


class CachedJinaScraper:
    
    def __init__(self, cache: SimpleCache):
        self.scraper = JinaWebScraper()
        self.cache = cache
    
    async def scrape_multiple(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple URLs with cache lookup"""
        results = []
        urls_to_scrape = []
        
        for url in urls:
            cached_content = self.cache.get_scrape(url)
            if cached_content:
                results.append({"url": url, "content": cached_content})
            else:
                urls_to_scrape.append(url)
        
        if urls_to_scrape:
            log_rag(f"Scraping {len(urls_to_scrape)} new URLs (found {len(results)} in cache)...")
            scraped_data = await self.scraper.scrape_multiple(urls_to_scrape)
            
            for item in scraped_data:
                self.cache.save_scrape(item['url'], item['content'])
                results.append(item)
        
        log_rag(f"Total: {len(results)} URLs available")
        return results