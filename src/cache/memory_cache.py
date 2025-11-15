# src/cache/memory_cache.py
"""Session-scoped in-memory cache without size limits.

UPDATES:
- Removed cache size limits for better performance
- Cache grows as needed during session
- Automatic cleanup on session end
"""

from src.search.google_search import GoogleSearcher
from src.search.jina_scraper import JinaWebScraper
from src.utils.logger import log_rag


class SimpleCache:
    """Session-based in-memory cache with automatic cleanup (no size limits)."""
    
    def __init__(self):
        self.search_cache = {}
        self.scrape_cache = {}
        log_rag("Cache initialized (unlimited size, session-based)")
    
    def get_search(self, query: str):
        """Check if we've searched for this before."""
        result = self.search_cache.get(query)
        
        if result:
            log_rag(f"Cache HIT for search: {query[:50]}...")
        
        return result
    
    def save_search(self, query: str, results: list):
        """Save search results to cache."""
        self.search_cache[query] = results
        log_rag(f"Cached search for: {query[:50]}...")
    
    def get_scrape(self, url: str):
        """Check if we've scraped this URL before."""
        return self.scrape_cache.get(url)
    
    def save_scrape(self, url: str, content: str):
        """Save scraped content to cache."""
        self.scrape_cache[url] = content
    
    def clear(self):
        """Clear all cache entries."""
        search_count = len(self.search_cache)
        scrape_count = len(self.scrape_cache)
        
        self.search_cache.clear()
        self.scrape_cache.clear()
        
        log_rag(f"Cache cleared ({search_count} searches, {scrape_count} scrapes)")
    
    def get_stats(self):
        """Get cache statistics."""
        return {
            "search_entries": len(self.search_cache),
            "scrape_entries": len(self.scrape_cache),
            "total_entries": len(self.search_cache) + len(self.scrape_cache)
        }
    
    def __del__(self):
        """Destructor - automatic cleanup on deletion."""
        try:
            self.clear()
        except:
            pass


class CachedGoogleSearcher:
    """Google searcher with session-scoped caching."""
    
    def __init__(self, api_key: str, cse_id: str, cache: SimpleCache):
        self.searcher = GoogleSearcher(api_key, cse_id)
        self.cache = cache
    
    async def search(self, query: str, num_results: int = 10):
        """Search with caching."""
        cached = self.cache.get_search(query)
        if cached:
            return cached[:num_results]
        
        results = await self.searcher.search(query, num_results)

        if results:
            self.cache.save_search(query, results)
        
        return results


class CachedJinaScraper:
    """Jina scraper with session-scoped caching."""
    
    def __init__(self, cache: SimpleCache):
        self.scraper = JinaWebScraper()
        self.cache = cache
    
    async def scrape_url(self, url: str):
        """Scrape a single URL with caching."""
        cached = self.cache.get_scrape(url)
        if cached:
            return cached
        
        content = await self.scraper.scrape_url(url)
        
        if content:
            self.cache.save_scrape(url, content)
        
        return content
    
    async def scrape_multiple(self, urls: list):
        """Scrape multiple URLs with caching."""
        results = []
        urls_to_scrape = []
        
        for url in urls:
            cached = self.cache.get_scrape(url)
            if cached:
                results.append({"url": url, "content": cached})
            else:
                urls_to_scrape.append(url)
        
        if urls_to_scrape:
            log_rag(f"Scraping {len(urls_to_scrape)} new URLs (rest from cache)...")
            scraped = await self.scraper.scrape_multiple(urls_to_scrape)
            
            for item in scraped:
                self.cache.save_scrape(item['url'], item['content'])
                results.append(item)
        
        log_rag(f"Total: {len(results)} URLs ({len(results) - len(urls_to_scrape)} from cache)")
        return results