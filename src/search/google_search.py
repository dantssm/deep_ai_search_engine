"""Google Custom Search API wrapper"""

import httpx
import asyncio
import random
from typing import List, Dict, Optional
from src.utils.logger import log_search


class GoogleSearcher:
    """
    Wrapper for the Google Custom Search JSON API.

    Attributes:
        api_key (str): Google API key.
        cse_id (str): Custom Search Engine ID.
        base_url (str): The Google API endpoint.
    """

    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id = cse_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    async def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Perform a Google search.

        Args:
            query (str): The search query.
            num_results (int): The target number of results to fetch.

        Returns:
            List[Dict]: A list of dictionaries representing search results.
                Each dictionary contains:
                - title (str): The title of the page.
                - link (str): The URL of the page.
                - snippet (str): A brief description/snippet.
        """
        async with httpx.AsyncClient() as client:
            results = []
            start_index = 1
            num_to_fetch = min(num_results, 100)

            while len(results) < num_to_fetch:
                response_data = await self._search_with_retry(client, query, start_index)
                
                if not response_data: break
                
                items = response_data.get("items", [])
                if not items: break
                
                for item in items:
                    results.append({
                        "title": item.get("title", ""), 
                        "link": item.get("link", ""), 
                        "snippet": item.get("snippet", "")
                    })
                    
                    if len(results) >= num_to_fetch: break
                
                start_index += 10
                await asyncio.sleep(0.1)
            
            log_search(f"Found {len(results)} results for '{query}'")
            return results
    
    async def _search_with_retry(self, client: httpx.AsyncClient, query: str, start: int) -> Optional[Dict]:
        """
        Execute a single search request with exponential backoff for rate limits.

        Args:
            client (httpx.AsyncClient): The HTTP client to use.
            query (str): The search term.
            start (int): The index of the first result to return (10 per page).

        Returns:
            Optional[Dict]: The JSON response from Google, or None if failed.
        """
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                params = {
                    "key": self.api_key, 
                    "cx": self.cse_id, 
                    "q": query, 
                    "num": 10, 
                    "start": start
                }
                
                response = await client.get(self.base_url, params=params, timeout=10.0)
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                # Rate limit
                if e.response.status_code == 429:
                    if attempt == max_attempts - 1:
                        log_search("Rate limit exhausted after max retries", level="error")
                        return None
                    
                    wait_time = (2 ** attempt) + random.uniform(0.1, 1.0)
                    log_search(f"Rate limit (429). Retrying in {wait_time:.1f}s...", level="warning")
                    await asyncio.sleep(wait_time)
                    continue

                log_search(f"Search failed: {e.response.status_code}", level="error")
                return None
            
            except Exception as e:
                log_search(f"Search error: {str(e)}", level="error")
                return None
        
        return None