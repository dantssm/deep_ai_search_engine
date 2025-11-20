"""
Researcher Graph Nodes.
Search -> Scrape & Index -> Retrieve -> Reflect -> (Repeat or Summarize)
"""

from typing import Dict
from urllib.parse import urlparse
from src.states import ResearcherState
from src.services import get_llm, LLMTier
from src.utils.logger import log_researcher
from src.prompts import get_reflection_prompt, get_summarization_prompt, format_context_chunks


class ResearcherError(Exception):
    pass


def calculate_source_quality(url: str) -> float:
    """
    Determine a trust score (0.0 - 1.0) based on the domain.
    
    Args:
        url (str): The source URL.
        
    Returns:
        float: Quality score.
    """
    domain = urlparse(url).netloc.lower()
    
    if any(d in domain for d in [".gov", ".edu", "wikipedia.org", "nih.gov", "nature.com"]):
        return 0.95
        
    if any(d in domain for d in ["github.com", "stackoverflow.com", "arxiv.org", "nytimes.com", "bbc.com"]):
        return 0.90
        
    if any(d in domain for d in ["medium.com", "linkedin.com", "reddit.com", "twitter.com", "x.com"]):
        return 0.60
        
    return 0.80


def is_valid_search_result(result: dict) -> bool:
    """
    Filter out broken links or empty results.
    
    Args:
        result (dict): A search result item with 'title', 'snippet', ...
        
    Returns:
        bool: True if the result looks useful.
    """
    text = (result.get("title", "") + " " + result.get("snippet", "")).lower()
    error_patterns = ["404 not found", "page not found", "access denied", "robot check"]
    
    if any(pattern in text for pattern in error_patterns):
        return False
    
    return len(text) >= 15


async def search_node(state: ResearcherState) -> Dict:
    """Search for information. Uses the 'next_query' from the previous reflection if available, otherwise uses the topic"""
    from src.services import get_searcher
    
    query = state["topic"]
    if state["reflections"]:
        last_reflection = state["reflections"][-1]
        if isinstance(last_reflection, dict) and last_reflection.get("next_query"):
            query = last_reflection["next_query"]
    
    log_researcher(f"Searching: {query[:50]}...")
    
    try:
        results = await get_searcher().search(query, num_results=10)
        
        valid_results = [r for r in results if is_valid_search_result(r)]
        
        if not valid_results:
            log_researcher("No valid results found.", level="warning")
            return {"searches": state["searches"] + [query]}

        new_sources = [
            {
                "url": r["link"], 
                "title": r["title"], 
                "snippet": r.get("snippet", "")
            }
            for r in valid_results[:5]
        ]
        
        return {
            "searches": state["searches"] + [query],
            "sources": state["sources"] + new_sources
        }
        
    except Exception as e:
        log_researcher(f"Search failed: {e}", level="error")
        return {"searches": state["searches"] + [query]}


async def scrape_and_index_node(state: ResearcherState) -> Dict:
    """Scrape URLs and save to Vector Store"""
    from src.services import get_scraper, get_rag_store
    
    current_urls = {s["url"] for s in state["sources"]}
    scraped_already = set(state["scraped_urls"])
    
    to_scrape = list(current_urls - scraped_already)[:5]
    
    if not to_scrape: return {}
    
    log_researcher(f"Scraping {len(to_scrape)} new URLs...")
    
    try:
        results = await get_scraper().scrape_multiple(to_scrape)
        
        valid_scrapes = [r for r in results if r and len(r.get("content", "")) > 100]
        
        quality_map = {}
        for item in valid_scrapes:
            url = item.get("url", "")
            quality_map[url] = calculate_source_quality(url)

        if valid_scrapes:
            store = get_rag_store()
            added_count = await store.add_documents(
                state["rag"]["collection_id"], 
                valid_scrapes,
                quality_scores=quality_map
            )
            log_researcher(f"Indexed {added_count} chunks into memory")
        
        return {
            "scraped_urls": state["scraped_urls"] + to_scrape,
            "rag": {**state["rag"], "chunks_indexed": state["rag"]["chunks_indexed"] + len(valid_scrapes)},
            "scraped_content": state.get("scraped_content", []) + valid_scrapes
        }
        
    except Exception as e:
        log_researcher(f"Scraping error: {e}", level="warning")
        return {"scraped_urls": state["scraped_urls"] + to_scrape}


async def retrieve_node(state: ResearcherState) -> Dict:
    """Retrieve information from the Vector Store"""
    from src.services import get_rag_store
    
    if state["rag"]["chunks_indexed"] == 0:
        log_researcher("Skipping retrieval (no data indexed)")
        return {"retrieved_chunks": []}
    
    try:
        store = get_rag_store()
        
        chunks = await store.search(
            state["rag"]["collection_id"], 
            state["topic"], 
            n=10,
            diversity_weight=0.3,
            use_mmr=True
        )
        return {"retrieved_chunks": chunks}
        
    except Exception as e:
        log_researcher(f"Retrieval error: {e}", level="error")
        return {"retrieved_chunks": []}


async def reflect_node(state: ResearcherState) -> Dict:
    """Analyzes what we found and decides: 'Do I know enough, or should I search again'?"""
    iter_count = state["iteration"] + 1
    
    if iter_count >= state["max_iterations"]:
        return {"iteration": iter_count}
    
    num_chunks = len(state["retrieved_chunks"])
    log_researcher(f"Reflecting on {num_chunks} chunks...")
    
    # if num_chunks >= 8:
    #     log_researcher("Sufficient content found (Heuristic). Stopping.")
    #     return {
    #         "reflections": state["reflections"] + [{
    #             "facts_learned": [],
    #             "gaps": [],
    #             "confidence": 0.8,
    #             "continue_research": False,
    #             "next_query": ""
    #         }],
    #         "iteration": iter_count
    #     }
    
    context = format_context_chunks(state["retrieved_chunks"])
    prompt = get_reflection_prompt(
        topic=state["topic"],
        parent_query=state.get("parent_query", state["topic"]),
        context=context,
        searches=state["searches"],
        num_chunks=num_chunks
    )
    
    try:
        llm = get_llm(LLMTier.FAST)
        decision_data = await llm.generate_json(prompt, max_tokens=500)
        
        if not isinstance(decision_data, dict):
            decision_data = {}
            
        decision_data.setdefault("confidence", 0.5)
        decision_data.setdefault("continue_research", False)
        decision_data.setdefault("next_query", "")
        
        if num_chunks < 5:
            decision_data['continue_research'] = True
            if not decision_data['next_query']:
                gaps = decision_data.get('gaps', [])
                suffix = gaps[0] if gaps else "overview"
                decision_data['next_query'] = f"{state['topic']} {suffix}"
        
        log_researcher(f"Decision: Continue={decision_data['continue_research']}, Conf={decision_data['confidence']:.2f}")
        
        return {
            "reflections": state["reflections"] + [decision_data],
            "iteration": iter_count,
            "gaps": state.get("gaps", []) + decision_data.get("gaps", [])
        }
        
    except Exception as e:
        log_researcher(f"Reflection failed: {e}", level="warning")
        return {"iteration": iter_count}


async def summarize_node(state: ResearcherState) -> Dict:
    """Summarize Findings. Compresses all research into a cited summary"""
    facts = []
    for r in state["reflections"]:
        if isinstance(r, dict):
            facts.extend(r.get("facts_learned", []))
    
    if not facts:
        facts = [c.get("content", "")[:150] for c in state["retrieved_chunks"][:8]]
    
    sources_list = []
    for i, source in enumerate(state["sources"], 1):
        title = source.get("title", "Untitled")[:60]
        sources_list.append(f"[{i}] {title}")
    
    prompt = get_summarization_prompt(
        topic=state["topic"],
        parent_query=state.get("parent_query", state["topic"]),
        facts=facts,
        sources=sources_list
    )
    
    try:
        llm = get_llm(LLMTier.FAST)
        summary = await llm.generate(prompt, max_tokens=1000)
        
        summary = summary.replace("Research Summary:", "").replace("Summary:", "").strip()
        
        num_chunks = len(state["retrieved_chunks"])
        if num_chunks >= 8:
            final_conf = 0.8
        elif num_chunks >= 5:
            final_conf = 0.6
        else:
            final_conf = 0.4
            
        return {
            "findings": summary,
            "quality_metrics": {
                "confidence": final_conf,
                "source_count": len(state["sources"]),
                "iterations_used": state["iteration"],
                "chunks_retrieved": num_chunks
            }
        }
        
    except Exception as e:
        log_researcher(f"Summarization error: {e}", level="error")
        return {
            "findings": f"Error summarizing research on {state['topic']}",
            "quality_metrics": {"confidence": 0}
        }


def should_continue(state: ResearcherState) -> str:
    """Decides direction: Search Again? or Finish?"""
    if state["iteration"] >= state["max_iterations"]:
        return "summarize"
    
    if not state["reflections"]:
        return "search"

    last_decision = state["reflections"][-1]
    if isinstance(last_decision, dict) and last_decision.get("continue_research", False):
        return "search"
    
    return "summarize"