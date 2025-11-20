"""Researcher graph nodes."""

from typing import Dict, List
from src.states import ResearcherState
from src.services import get_llm, LLMTier
from src.utils.logger import log_researcher
from src.prompts import (
    get_reflection_prompt,
    get_summarization_prompt,
    format_context_chunks
)


class ResearcherError(Exception):
    """Raised when researcher encounters unrecoverable error."""
    pass


def is_valid_search_result(result: dict) -> bool:
    """Check if search result is valid."""
    text = (result.get("title", "") + " " + result.get("snippet", "")).lower()
    error_patterns = ["404 not found", "page not found", "access denied"]
    
    if any(pattern in text for pattern in error_patterns):
        return False
    
    return len(text) >= 15


async def search_node(state: ResearcherState) -> dict:
    """Search for information on the topic."""
    from src.services import get_searcher
    
    query = state["topic"]
    if state["reflections"]:
        last = state["reflections"][-1]
        if isinstance(last, dict) and last.get("next_query"):
            query = last["next_query"]
    
    log_researcher(f"Search: {query[:50]}")
    
    try:
        results = await get_searcher().search(query, num_results=10)
        valid_results = [r for r in results if is_valid_search_result(r)]
        
        if len(valid_results) < len(results):
            log_researcher(f"Filtered {len(results)} to {len(valid_results)} results")
        
        if not valid_results:
            log_researcher("No relevant results", level="warning")
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
        log_researcher(f"Search error: {e}", level="error")
        raise ResearcherError(str(e))


async def scrape_and_index_node(state: ResearcherState) -> dict:
    """Scrape URLs and index content in RAG store."""
    from src.services import get_scraper, get_rag_store
    
    current_urls = {s["url"] for s in state["sources"]}
    scraped_already = set(state["scraped_urls"])
    to_scrape = list(current_urls - scraped_already)[:5]
    
    if not to_scrape:
        return {}
    
    log_researcher(f"Scraping {len(to_scrape)} URLs...")
    
    try:
        results = await get_scraper().scrape_multiple(to_scrape)
        valid_scrapes = [r for r in results if r and len(r.get("content", "")) > 100]
        
        if valid_scrapes:
            store = get_rag_store()
            added = await store.add_documents(state["rag"]["collection_id"], valid_scrapes)
            log_researcher(f"Indexed {added} chunks")
        
        return {
            "scraped_urls": state["scraped_urls"] + to_scrape,
            "rag": {
                **state["rag"], 
                "chunks_indexed": state["rag"]["chunks_indexed"] + len(valid_scrapes)
            },
            "scraped_content": state.get("scraped_content", []) + valid_scrapes
        }
    except Exception as e:
        log_researcher(f"Scrape error: {e}", level="warning")
        return {"scraped_urls": state["scraped_urls"] + to_scrape}


async def retrieve_node(state: ResearcherState) -> dict:
    """Retrieve relevant chunks from RAG store."""
    from src.services import get_rag_store
    
    if state["rag"]["chunks_indexed"] == 0:
        return {"retrieved_chunks": []}
    
    try:
        store = get_rag_store()
        chunks = await store.search(
            state["rag"]["collection_id"], 
            state["topic"], 
            n=10,
            diversity_weight=0.3,
            use_query_expansion=True,
            use_mmr=True
        )
        return {"retrieved_chunks": chunks}
    except Exception as e:
        log_researcher(f"Retrieve error: {e}", level="error")
        return {"retrieved_chunks": []}


async def reflect_node(state: ResearcherState) -> dict:
    """Reflect on progress and decide next steps."""
    iter_count = state["iteration"] + 1
    
    if iter_count >= state["max_iterations"]:
        return {"iteration": iter_count}
    
    num_chunks = len(state["retrieved_chunks"])
    log_researcher(f"Retrieved {num_chunks} chunks")
    
    if num_chunks >= 8:
        log_researcher("Sufficient content retrieved, stopping early")
        return {
            "reflections": state["reflections"] + [{
                "facts_learned": [c.get("content", "")[:100] for c in state["retrieved_chunks"][:5]],
                "gaps": [],
                "confidence": 0.7,
                "continue_research": False,
                "next_query": ""
            }],
            "iteration": iter_count
        }
    
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
        data = await llm.generate_json(prompt, max_tokens=500)
        
        if not isinstance(data, dict):
            raise ValueError("Response is not a dictionary")
        
        data.setdefault("facts_learned", [])
        data.setdefault("gaps", [])
        data.setdefault("confidence", 0.5)
        data.setdefault("continue_research", False)
        data.setdefault("next_query", "")
        
        if num_chunks >= 6:
            data['confidence'] = max(data.get('confidence', 0.5), 0.65)
        
        if num_chunks < 5:
            data['continue_research'] = True
            if not data.get('next_query'):
                gaps = data.get('gaps', [])
                if gaps:
                    data['next_query'] = f"{state['topic']} {gaps[0]}"
        else:
            data['continue_research'] = False
        
        log_researcher(f"Reflection: confidence={data.get('confidence', 0):.2f}, continue={data.get('continue_research', False)}")
        
        return {
            "reflections": state["reflections"] + [data],
            "iteration": iter_count,
            "gaps": state.get("gaps", []) + data.get("gaps", [])
        }
        
    except Exception as e:
        log_researcher(f"Reflection failed: {e}", level="warning")
        return {
            "reflections": state["reflections"] + [{
                "confidence": 0.55,
                "continue_research": False,
                "gaps": [],
                "facts_learned": [],
                "next_query": ""
            }],
            "iteration": iter_count
        }


async def summarize_node(state: ResearcherState) -> dict:
    """Compile final findings with quality metrics."""
    facts = []
    for r in state["reflections"]:
        if isinstance(r, dict):
            facts.extend(r.get("facts_learned", []))
    
    if not facts:
        facts = [c.get("content", "")[:100] for c in state["retrieved_chunks"][:5]]
    
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
        summary = await llm.generate(prompt, max_tokens=500)
        
        summary = summary.replace("Research Summary:", "").replace("Summary:", "").strip()
        summary = summary.replace("According to the summary,", "").strip()
        summary = summary.replace("The research summary", "Research").strip()
        
        num_chunks = len(state["retrieved_chunks"])
        
        if num_chunks >= 8:
            final_confidence = 0.70
        elif num_chunks >= 6:
            final_confidence = 0.65
        elif num_chunks >= 5:
            final_confidence = 0.60
        else:
            reflections = [r for r in state["reflections"] if isinstance(r, dict)]
            confidences = [r.get("confidence", 0) for r in reflections]
            final_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        return {
            "findings": summary,
            "quality_metrics": {
                "confidence": final_confidence,
                "source_count": len(state["sources"]),
                "iterations_used": state["iteration"],
                "chunks_retrieved": num_chunks
            }
        }
    except Exception as e:
        log_researcher(f"Summarization error: {e}", level="error")
        return {
            "findings": "Error: Unable to generate summary",
            "quality_metrics": {"confidence": 0}
        }


def should_continue(state: ResearcherState) -> str:
    """Decide whether to continue researching or summarize."""
    if state["iteration"] >= state["max_iterations"]:
        return "summarize"
    
    if not state["reflections"]:
        return "search"
    
    last = state["reflections"][-1]
    
    if isinstance(last, dict) and last.get("continue_research", False):
        return "search"
    
    return "summarize"