"""Google Custom Search API wrapper"""

import httpx
import asyncio
import random
from typing import List, Dict
from src.utils.logger import log_search


class GoogleSearcher:
    """Google Custom Search API client"""

    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id = cse_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    async def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Perform Google search with retry on rate limits"""
        async with httpx.AsyncClient() as client:
            results = []
            start_index = 1
            num_to_fetch = min(num_results, 100)

            while len(results) < num_to_fetch:
                response = await self._search_with_retry(client, query, start_index)
                
                if response is None:
                    break
                
                items = response.get("items", [])
                if not items:
                    break
                
                for item in items:
                    results.append({
                        "title": item.get("title", ""), 
                        "link": item.get("link", ""), 
                        "snippet": item.get("snippet", "")
                    })
                    
                    if len(results) >= num_to_fetch:
                        break
                
                start_index += 10
                await asyncio.sleep(0.1)
            
            log_search(f"Found {len(results)} results for '{query}'")
            return results
    
    async def _search_with_retry(self, client: httpx.AsyncClient, query: str, start: int) -> dict:
        """Execute single search request with exponential backoff on 429."""
        for attempt in range(3):
            try:
                response = await client.get(
                    self.base_url, 
                    params={
                        "key": self.api_key, 
                        "cx": self.cse_id, 
                        "q": query, 
                        "num": 10, 
                        "start": start
                    }, 
                    timeout=10.0
                )
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 429:
                    log_search(f"Search failed: {e.response.status_code}", level="error")
                    return None
                
                if attempt == 2:
                    log_search("Rate limit exhausted after 3 attempts", level="error")
                    return None
                
                wait = (2 ** attempt) + random.uniform(0.1, 1.0)
                log_search(f"Rate limit (429). Retrying in {wait:.1f}s...", level="warning")
                await asyncio.sleep(wait)
            
            except Exception as e:
                log_search(f"Search error: {str(e)}", level="error")
                return None
        
        return None