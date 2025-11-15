import httpx
import asyncio
import random
from typing import List, Dict
from src.utils.logger import log_search

class GoogleSearcher:

    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id = cse_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    async def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Performs a Google search with 429 retry logic.
        """
        async with httpx.AsyncClient() as client:
            results = []
            start_index = 1
            num_to_fetch = min(num_results, 100)

            while len(results) < num_to_fetch:
                try:
                    response = None
                    last_error = None
                    
                    for attempt in range(3):
                        try:
                            response = await client.get(
                                self.base_url, 
                                params={
                                    "key": self.api_key, 
                                    "cx": self.cse_id, 
                                    "q": query, 
                                    "num": 10, 
                                    "start": start_index
                                }, 
                                timeout=10.0
                            )
                            response.raise_for_status()
                            break
                        except httpx.HTTPStatusError as e:
                            if e.response.status_code == 429:
                                wait = (2 ** attempt) + random.uniform(0.1, 1.0)
                                log_search(f"Rate limit (429). Retrying in {wait:.1f}s...", level="warning")
                                await asyncio.sleep(wait)
                                last_error = e
                            else:
                                raise e
                        except Exception as e:
                            last_error = e
                            break
                            
                    if response is None:
                        if last_error: raise last_error
                        break

                    data = response.json()
                    
                    items = data.get("items", [])
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
                
                except httpx.HTTPStatusError as e:
                    log_search(f"Search failed: {e.response.status_code}", level="error")
                    break
                
                except Exception as e:
                    log_search(f"Search error: {str(e)}", level="error")
                    break
            
            log_search(f"Found {len(results)} results for '{query}'")
            return results