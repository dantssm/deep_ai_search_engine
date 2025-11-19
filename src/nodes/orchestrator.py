import asyncio
import hashlib
import re
from typing import List, Dict, Set
from collections import Counter
from src.states import OrchestratorState, ResearcherState
from src.services import get_llm, LLMTier
from src.utils.logger import log_orchestrator
from src.prompts import (
    get_followup_topics_prompt,
    get_synthesis_prompt,
    format_sources_for_synthesis,
    format_findings_by_topic
)

def deduplicate_sources(sources: List[Dict]) -> List[Dict]:
    """Remove duplicate sources by URL."""
    seen_urls = set()
    unique_sources = []
    
    for source in sources:
        url = source.get("url", "")
        if "#" in url:
            url = url.split("#")[0]
        url = url.rstrip("/")
        
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(source)
    
    log_orchestrator(f"Deduplicated {len(sources)} to {len(unique_sources)} sources")
    return unique_sources


def calculate_source_diversity(sources: List[Dict]) -> float:
    """Calculate diversity score based on unique domains."""
    from urllib.parse import urlparse
    
    domains = set()
    for source in sources:
        url = source.get("url", "")
        try:
            domain = urlparse(url).netloc
            if domain:
                domains.add(domain)
        except:
            pass
    
    return len(domains) / max(len(sources), 1)


def extract_all_citations(text: str) -> Set[int]:
    """Extract all citation IDs from text, handling both [1] and [1, 2, 3] formats.
    
    Examples:
        "[1]" -> {1}
        "[1, 2, 3]" -> {1, 2, 3}
        "[1][2][3]" -> {1, 2, 3}
        "text [1, 2] more [3]" -> {1, 2, 3}
    
    Returns:
        Set of unique citation IDs
    """
    citation_ids = set()
    
    citation_patterns = re.findall(r'\[([0-9,\s]+)\]', text)
    
    for pattern in citation_patterns:
        ids = pattern.split(',')
        for id_str in ids:
            id_str = id_str.strip()
            if id_str.isdigit():
                citation_ids.add(int(id_str))
    
    return citation_ids


async def generate_followup_topics(query: str, completed: List[Dict], 
                                   gaps: List[str], avg_confidence: float) -> List[str]:
    """Generate new topics to fill research gaps."""
    prompt = get_followup_topics_prompt(query, completed, gaps, avg_confidence)
    
    try:
        llm = get_llm(LLMTier.SMART)
        response = await llm.generate(prompt)
        new_topics = [line.strip().lstrip("- ") for line in response.strip().split("\n") if line.strip()]
        return new_topics[:3]
    except Exception as e:
        log_orchestrator(f"Deepening failed: {e}", level="warning")
        return []

async def run_single_researcher(topic: str, shared_context: str, 
                                global_scraped: Set[str]) -> dict:
    """Run a single researcher on a topic."""
    from src.graphs import build_researcher_graph
    
    researcher = build_researcher_graph()
    coll_id = f"res_{hashlib.md5(topic.encode()).hexdigest()[:10]}"
    
    initial_state: ResearcherState = {
        "topic": topic,
        "parent_query": shared_context,
        "searches": [],
        "scraped_urls": list(global_scraped),
        "rag": {"collection_id": coll_id, "chunks_indexed": 0},
        "reflections": [],
        "iteration": 0,
        "max_iterations": 2,
        "scraped_content": [],
        "retrieved_chunks": [],
        "findings": "",
        "sources": [],
        "gaps": [],
        "quality_metrics": None
    }
    
    result = await researcher.ainvoke(initial_state)
    return result

async def plan_node(state: OrchestratorState) -> dict:
    """Display research plan and initialize tracking."""
    log_orchestrator("=" * 60)
    log_orchestrator("RESEARCH PLAN")
    log_orchestrator("=" * 60)
    log_orchestrator(f"Query: {state['query']}")
    log_orchestrator(f"\nSub-topics ({len(state['sub_topics'])}):")
    for i, topic in enumerate(state['sub_topics'], 1):
        log_orchestrator(f"   {i}. {topic}")
    log_orchestrator("=" * 60)
    
    return {
        "retry_queue": [],
        "global_scraped_urls": set(),
        "failed_topics": set()
    }


async def dispatch_node(state: OrchestratorState) -> dict:
    """Dispatch researchers with retry and deduplication logic."""
    log_orchestrator("=" * 60)
    log_orchestrator(f"DISPATCH (iteration {state['iteration'] + 1})")
    log_orchestrator("=" * 60)
    
    completed_topics = {r["topic"] for r in state["completed"]}
    failed_topics = state.get("failed_topics", set())
    
    pending_topics = [
        t for t in state["sub_topics"] 
        if t not in completed_topics and t not in failed_topics
    ]

    retry_topics = []
    current_iter = state.get("iteration", 0)
    new_retry_queue = []
    
    for retry in state.get("retry_queue", []):
        wait_time = 2 ** (retry["attempt"] - 1)
        if current_iter - retry.get("last_attempt_at", 0) >= wait_time:
            if retry["attempt"] < 3:
                retry_topics.append(retry)
            else:
                failed_topics.add(retry["topic"])
                log_orchestrator(f"Permanently failed: {retry['topic'][:40]}")
        else:
            new_retry_queue.append(retry)
    
    batch_size = 3
    tasks_to_run = []
    
    for r in retry_topics[:batch_size]:
        tasks_to_run.append({"topic": r["topic"], "is_retry": True})
    
    remaining_slots = batch_size - len(tasks_to_run)
    for t in pending_topics[:remaining_slots]:
        tasks_to_run.append({"topic": t, "is_retry": False})
    
    if not tasks_to_run:
        log_orchestrator("No topics ready to research", level="warning")
        return {"failed_topics": failed_topics, "retry_queue": new_retry_queue}
    
    log_orchestrator(f"Researching {len(tasks_to_run)} topics:")
    for task in tasks_to_run:
        prefix = 'RETRY' if task['is_retry'] else 'NEW'
        log_orchestrator(f"   [{prefix}] {task['topic'][:60]}")
    
    shared_context = "\n".join([
        f"- {r['topic']}: {r['findings'][:200]}..."
        for r in state["completed"][-3:]
    ]) if state["completed"] else "(No prior research yet)"
    
    full_context = f"{state['query']}\n\nContext:\n{shared_context}"
    global_scraped = state.get("global_scraped_urls", set())
    
    coroutines = [
        run_single_researcher(task["topic"], full_context, global_scraped)
        for task in tasks_to_run
    ]
    
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    
    new_completed = []
    new_sources = []
    new_scraped_urls = set(global_scraped)
    
    for i, result in enumerate(results):
        topic = tasks_to_run[i]["topic"]
        
        if isinstance(result, Exception):
            error_str = str(result)
            is_retriable = "timeout" in error_str.lower() or "429" in error_str
            
            if is_retriable:
                existing = next((r for r in retry_topics if r["topic"] == topic), None)
                attempt = (existing["attempt"] + 1) if existing else 1
                
                new_retry_queue.append({
                    "topic": topic,
                    "attempt": attempt,
                    "last_error": error_str,
                    "last_attempt_at": state.get("iteration", 0)
                })
                log_orchestrator(f"Re-queueing: {topic[:30]} (attempt {attempt})")
            else:
                failed_topics.add(topic)
                log_orchestrator(f"Failed: {topic[:30]} - {error_str[:30]}")
            continue
        
        metrics = result.get("quality_metrics", {})
        log_orchestrator(f"Completed: {topic[:30]} (Confidence: {metrics.get('confidence', 0):.2f})")
        
        new_completed.append({
            "topic": result["topic"],
            "findings": result["findings"],
            "sources": result["sources"],
            "gaps": result["gaps"],
            "quality_metrics": metrics
        })
        new_sources.extend(result["sources"])
        new_scraped_urls.update(result.get("scraped_urls", []))
    
    return {
        "completed": state["completed"] + new_completed,
        "all_sources": state["all_sources"] + new_sources,
        "retry_queue": new_retry_queue,
        "global_scraped_urls": new_scraped_urls,
        "failed_topics": failed_topics,
        "iteration": state["iteration"] + 1
    }


async def critique_node(state: OrchestratorState) -> dict:
    """Analyze results and generate new topics if confidence is low.
    
    BALANCED: Confidence threshold at 0.60 (middle ground between 0.55 and 0.70).
    Allows up to 4 iterations for better depth.
    """
    log_orchestrator("=" * 60)
    log_orchestrator("CRITIQUE & DEEPENING")
    log_orchestrator("=" * 60)
    
    completed = state["completed"]
    if not completed:
        return {}

    confidences = [r.get("quality_metrics", {}).get("confidence", 0) for r in completed]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    log_orchestrator(f"Current Average Confidence: {avg_confidence:.2f}")
    
    if avg_confidence < 0.60 and state["iteration"] < min(state["max_iterations"], 4):
        log_orchestrator("Confidence below threshold (0.60). Generating follow-up topics...")
        
        all_gaps = []
        for r in completed:
            all_gaps.extend(r.get("gaps", []))
        
        unique_gaps = list(set(all_gaps))[:5]
        
        new_topics = await generate_followup_topics(
            state['query'], 
            completed, 
            unique_gaps, 
            avg_confidence
        )
        
        existing = set(state["sub_topics"])
        final_new = [t for t in new_topics if t not in existing][:2]
        
        if final_new:
            log_orchestrator(f"Added {len(final_new)} new topics for deepening:")
            for t in final_new:
                log_orchestrator(f"   > {t}")
            
            return {"sub_topics": state["sub_topics"] + final_new}
        else:
            log_orchestrator("No unique new topics generated.")

    return {}


async def synthesize_node(state: OrchestratorState) -> dict:
    """Final synthesis with structured JSON output and improved citation extraction."""
    log_orchestrator("=" * 60)
    log_orchestrator("FINAL SYNTHESIS")
    log_orchestrator("=" * 60)
    
    unique_sources = deduplicate_sources(state["all_sources"])
    
    for i, source in enumerate(unique_sources, 1):
        source["id"] = i
    
    findings_text = format_findings_by_topic(state["completed"])
    source_list = format_sources_for_synthesis(unique_sources)
    
    prompt = get_synthesis_prompt(
        query=state["query"],
        findings_by_topic=findings_text,
        source_list=source_list
    )
    
    try:
        llm = get_llm(LLMTier.SMART)
        
        report_text = await llm.generate(prompt, max_tokens=8000)
        
        if len(report_text) < 500:
            log_orchestrator("Report suspiciously short, may be truncated", level="warning")
        
        cited_source_ids = sorted(extract_all_citations(report_text))
        
        log_orchestrator(f"Extracted {len(cited_source_ids)} unique citations: {cited_source_ids}")
        
        confidences = [r.get("quality_metrics", {}).get("confidence", 0.5) for r in state["completed"]]
        final_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Build structured result
        synthesis_result = {
            "report_text": report_text,
            "sources_used": unique_sources,
            "citations": cited_source_ids,
            "metadata": {
                "confidence": final_confidence,
                "source_count": len(unique_sources),
                "sources_cited": len(cited_source_ids)
            }
        }
        
        log_orchestrator(f"Synthesis complete (confidence: {final_confidence:.2f}, {len(cited_source_ids)} sources cited)")
        
        return {
            "synthesis_result": synthesis_result,
            "report": report_text,
            "overall_quality": synthesis_result["metadata"]
        }
        
    except Exception as e:
        log_orchestrator(f"Synthesis failed: {e}", level="error")
        
        fallback_report = f"# {state['query']}\n\n"
        for r in state["completed"]:
            fallback_report += f"## {r['topic']}\n\n{r['findings']}\n\n"
        
        return {
            "synthesis_result": {
                "report_text": fallback_report,
                "sources_used": unique_sources,
                "citations": [],
                "metadata": {
                    "confidence": 0.3,
                    "source_count": len(unique_sources),
                    "sources_cited": 0
                }
            },
            "report": fallback_report
        }


def should_continue_orchestrator(state: OrchestratorState) -> str:
    """Check if we have more work (including newly added topics)."""
    completed = {r["topic"] for r in state["completed"]}
    failed = state.get("failed_topics", set())
    
    pending = [t for t in state["sub_topics"] if t not in completed and t not in failed]
    retries = state.get("retry_queue", [])
    
    if (pending or retries) and state["iteration"] < state["max_iterations"]:
        log_orchestrator(f"Next: DISPATCH ({len(pending)} pending, {len(retries)} retries)")
        return "dispatch"
    
    log_orchestrator("Next: SYNTHESIZE (Research Complete)")
    return "synthesize"