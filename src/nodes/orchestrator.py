"""
Orchestrator Graph Nodes.

1. Initialize tracking.
2. Assign sub-topics to Researcher agents.
3. Evaluate if the research is sufficient or if we need deep-dives.
4. Combine all findings into a final report.
"""

import asyncio
import hashlib
import re
from typing import Set
from src.states import OrchestratorState, ResearcherState
from src.services import get_llm, LLMTier
from src.utils.logger import log_orchestrator
from src.prompts import get_followup_topics_prompt, get_synthesis_prompt, format_sources_for_synthesis


def extract_citation_ids(text: str) -> Set[int]:
    """Extract unique citation numbers [1], [2] from text."""
    ids = set()
    for pattern in re.findall(r'\[([0-9,\s]+)\]', text):
        for part in pattern.split(','):
            if part.strip().isdigit():
                ids.add(int(part.strip()))
    return ids

async def run_single_researcher(topic: str, shared_context: str, global_scraped: Set[str]) -> dict:
    """
    Run an isolated researcher graph for a specific topic.
    
    Args:
        topic (str): The specific question to research.
        shared_context (str): Context from previous agents to avoid duplication.
        global_scraped (Set[str]): URLs already visited by other agents.    
    """
    from src.graphs import build_researcher_graph
    
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
    
    return await build_researcher_graph().ainvoke(initial_state)

async def plan_node(state: OrchestratorState) -> dict:
    """Initialize the research session tracking"""
    log_orchestrator(f"Planning: {state['query']}")
    
    for i, topic in enumerate(state['sub_topics'], 1):
        log_orchestrator(f"    {i}. {topic}")
    
    return {
        "retry_queue": [],
        "global_scraped_urls": set(),
        "failed_topics": set()
    }

async def dispatch_node(state: OrchestratorState) -> dict:
    """Determine which topics to research next"""
    iteration = state.get("iteration", 0)
    log_orchestrator(f"Dispatch (Iter {iteration + 1})")
    
    completed_topics = {r["topic"] for r in state["completed"]}
    failed_topics = state.get("failed_topics", set())
    
    pending = [t for t in state["sub_topics"] if t not in completed_topics and t not in failed_topics]
    
    ready_retries = []
    future_retries = []
    
    for item in state.get("retry_queue", []):
        wait_time = 2 ** (item["attempt"] - 1)
        if iteration - item.get("last_attempt_at", 0) >= wait_time:
            if item["attempt"] < 3:
                ready_retries.append(item)
            else:
                failed_topics.add(item["topic"])
                log_orchestrator(f"Giving up on topic: {item['topic'][:30]}. Reached max retries.")
        else:
            future_retries.append(item)
    
    BATCH_SIZE = 3
    tasks_to_run = []
    
    for r in ready_retries[:BATCH_SIZE]:
        tasks_to_run.append({"topic": r["topic"], "is_retry": True})
    
    slots_left = BATCH_SIZE - len(tasks_to_run)
    for t in pending[:slots_left]:
        tasks_to_run.append({"topic": t, "is_retry": False})
    
    if not tasks_to_run:
        return {"failed_topics": failed_topics, "retry_queue": future_retries}
    
    context_snippets = [f"- {r['topic']}: {r['findings'][:150]}..." for r in state["completed"][-3:]]
    shared_context = f"Goal: {state['query']}\nPrevious Findings:\n" + "\n".join(context_snippets)
    
    global_scraped = state.get("global_scraped_urls", set())
    
    coroutines = [
        run_single_researcher(t["topic"], shared_context, global_scraped)
        for t in tasks_to_run
    ]
    
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    
    new_completed = []
    new_sources = []
    new_scraped = set(global_scraped)
    updated_retry_queue = list(future_retries)
    
    for i, result in enumerate(results):
        task = tasks_to_run[i]
        topic = task["topic"]
        
        if isinstance(result, Exception):
            error_msg = str(result)
            is_retriable = "timeout" in error_msg.lower() or "429" in error_msg
            
            if is_retriable:
                existing_retry = next((r for r in ready_retries if r["topic"] == topic), None)
                attempt = (existing_retry["attempt"] + 1) if existing_retry else 1
                
                updated_retry_queue.append({
                    "topic": topic,
                    "attempt": attempt,
                    "last_attempt_at": iteration,
                    "last_error": error_msg
                })
                log_orchestrator(f"Re-queueing {topic[:20]} (Attempt {attempt})")
            else:
                failed_topics.add(topic)
                log_orchestrator(f"Failed {topic[:20]}: {error_msg}")
            continue
        
        metrics = result.get("quality_metrics", {})
        log_orchestrator(f"Finished: {topic[:30]} (Conf: {metrics.get('confidence', 0):.2f})")
        
        new_completed.append({
            "topic": result["topic"],
            "findings": result["findings"],
            "sources": result["sources"],
            "gaps": result["gaps"],
            "quality_metrics": metrics
        })
        new_sources.extend(result["sources"])
        new_scraped.update(result.get("scraped_urls", []))
        
    return {
        "completed": state["completed"] + new_completed,
        "all_sources": state["all_sources"] + new_sources,
        "retry_queue": updated_retry_queue,
        "global_scraped_urls": new_scraped,
        "failed_topics": failed_topics,
        "iteration": iteration + 1
    }

async def critique_node(state: OrchestratorState) -> dict:
    """
    Evaluate if we need to deepen the research.
    Trigger condition: Average confidence < 0.6
    """
    completed = state["completed"]
    if not completed:
        return {}

    confidences = [r.get("quality_metrics", {}).get("confidence", 0) for r in completed]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    if avg_confidence < 0.60 and state["iteration"] < min(state["max_iterations"], 4):
        log_orchestrator(f"Low Confidence ({avg_confidence:.2f}). Generating follow-up topics...")
        
        all_gaps = []
        for r in completed:
            all_gaps.extend(r.get("gaps", []))
        
        prompt = get_followup_topics_prompt(state['query'], completed, list(set(all_gaps))[:5], avg_confidence)
        try:
            llm = get_llm(LLMTier.SMART)
            response = await llm.generate(prompt)
            new_topics = [t.strip().lstrip("- ") for t in response.split("\n") if t.strip()]
            
            existing = set(state["sub_topics"])
            final_new = [t for t in new_topics if t not in existing][:2]
            
            if final_new:
                log_orchestrator(f"Added {len(final_new)} new topics")
                return {"sub_topics": state["sub_topics"] + final_new}
                
        except Exception as e:
            log_orchestrator(f"Deepening failed: {e}", level="warning")

    return {}

async def synthesize_node(state: OrchestratorState) -> dict:
    """Compile final report. Re-maps local citations to global citations across all topics"""
    log_orchestrator("Synthesizing report")
    
    global_sources = []
    url_to_id = {}
    formatted_findings = []
    
    for result in state["completed"]:
        local_to_global = {}
        
        for local_source in result["sources"]:
            url = local_source.get("url", "").split("#")[0].rstrip("/")
            
            if url not in url_to_id:
                global_sources.append(local_source)
                new_id = len(global_sources)
                url_to_id[url] = new_id
                local_source["id"] = new_id
            
            local_idx = result["sources"].index(local_source) + 1
            local_to_global[local_idx] = url_to_id[url]
        
        findings = result["findings"]
        for local_id in sorted(local_to_global.keys(), reverse=True):
            global_id = local_to_global[local_id]
            findings = re.sub(rf'\[{local_id}\]', f'[{global_id}]', findings)
            
        formatted_findings.append(f"Topic: {result['topic']}\n{findings}")

    source_list_text = format_sources_for_synthesis(global_sources)
    prompt = get_synthesis_prompt(
        query=state["query"],
        findings_by_topic='\n\n'.join(formatted_findings),
        source_list=source_list_text
    )
    
    try:
        llm = get_llm(LLMTier.SMART)
        report_text = await llm.generate(prompt, max_tokens=8000)
        
        cited_ids = sorted(extract_citation_ids(report_text))

        base_conf = sum(r.get("quality_metrics", {}).get("confidence", 0.5) for r in state["completed"])
        avg_conf = base_conf / len(state["completed"]) if state["completed"] else 0.5
        
        synthesis_result = {
            "report_text": report_text,
            "sources_used": global_sources,
            "citations": cited_ids,
            "metadata": {
                "confidence": avg_conf,
                "source_count": len(global_sources),
                "sources_cited": len(cited_ids)
            }
        }
        
        return {
            "synthesis_result": synthesis_result,
            "report": report_text
        }
        
    except Exception as e:
        log_orchestrator(f"Synthesis failed: {e}", level="error")
        return {"report": "\n".join(formatted_findings)}

def should_continue_orchestrator(state: OrchestratorState) -> str:
    """Dispatch or synthesize?"""
    completed = {r["topic"] for r in state["completed"]}
    failed = state.get("failed_topics", set())
    
    pending = [t for t in state["sub_topics"] if t not in completed and t not in failed]
    retries = state.get("retry_queue", [])
    
    if (pending or retries) and state["iteration"] < state["max_iterations"]:
        return "dispatch"
    
    return "synthesize"