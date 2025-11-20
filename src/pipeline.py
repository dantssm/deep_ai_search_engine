"""Main research pipeline with session isolation and JSON-based results."""

from typing import Dict, Optional, Callable, Awaitable
from datetime import datetime
from src.states import OrchestratorState
from src.graphs import build_orchestrator_graph
from src.services import get_llm, get_rag_store, set_current_session, LLMTier
from src.config import DEPTH_PARAMS
from src.utils.logger import log_pipeline
from src.prompts import (
    get_topic_breakdown_prompt,
    get_reasoning_prompt,
    get_refinement_prompt
)
import asyncio


class DeepResearchPipeline:
    """Session-scoped research pipeline."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.graph = build_orchestrator_graph()
        self.last_result = None
        
        log_pipeline(f"Pipeline created for session {session_id[:8]}")
    
    async def create_plan(self, query: str, depth: str) -> Dict:
        """Generate initial research plan."""
        set_current_session(self.session_id)
        
        params = DEPTH_PARAMS.get(depth, DEPTH_PARAMS["standard"])
        num_topics = params.get("max_searches", 3)
        
        log_pipeline(f"Creating plan for: '{query}' (depth: {depth})")
        
        llm = get_llm(LLMTier.SMART)

        topics_prompt = get_topic_breakdown_prompt(query, num_topics)
        response = await llm.generate(topics_prompt)
        sub_topics = [line.strip() for line in response.strip().split("\n") if line.strip()][:num_topics]
        
        if not sub_topics:
            log_pipeline("No topics generated, using query as single topic", level="warning")
            sub_topics = [query]
        
        log_pipeline(f"Generated {len(sub_topics)} sub-topics")

        reasoning_prompt = get_reasoning_prompt(query)
        reasoning_response = await llm.generate(reasoning_prompt)
        
        return {
            "query": query,
            "depth": depth,
            "sub_topics": sub_topics,
            "reasoning": reasoning_response.strip(),
            "estimated_sources": len(sub_topics) * 5
        }
    
    async def refine_plan(self, query: str, depth: str, current_plan: Dict, feedback: str) -> Dict:
        """Refine plan based on user feedback."""
        set_current_session(self.session_id)
        
        log_pipeline(f"Refining plan based on feedback: '{feedback[:50]}...'")
        
        llm = get_llm(LLMTier.SMART)
        
        current_topics = current_plan.get("sub_topics", [])
        num_topics = len(current_topics)
        
        prompt = get_refinement_prompt(query, current_topics, feedback, num_topics)
        response = await llm.generate(prompt)
        new_topics = [line.strip() for line in response.strip().split("\n") if line.strip()]
        
        if not new_topics:
            log_pipeline("Refinement produced no topics, keeping current", level="warning")
            new_topics = current_topics
        
        log_pipeline(f"Refined to {len(new_topics)} topics")
        
        new_reasoning = f"Refined based on feedback: {feedback}"
        
        return {
            **current_plan,
            "sub_topics": new_topics[:num_topics],
            "reasoning": new_reasoning
        }
    
    async def execute_research(
        self, 
        plan: Dict,
        on_progress: Optional[Callable[[str], Awaitable]] = None
    ) -> Dict:
        """Execute research and return structured JSON."""
        set_current_session(self.session_id)
        
        max_iters = DEPTH_PARAMS.get(plan["depth"], DEPTH_PARAMS["standard"]).get("max_searches", 3)
        
        log_pipeline(f"Executing research with {len(plan['sub_topics'])} topics")
        
        init_state: OrchestratorState = {
            "query": plan["query"],
            "depth": plan["depth"],
            "sub_topics": plan["sub_topics"],
            "iteration": 0,
            "max_iterations": max_iters,
            "completed": [],
            "all_sources": [],
            "identified_gaps": [],
            "report": "",
            "synthesis_result": None
        }
        
        if on_progress:
            await on_progress("Researching sub-topics...")
        
        result = await self.graph.ainvoke(init_state)
        
        if on_progress:
            await on_progress("Finalizing report...")
        
        synthesis = result.get("synthesis_result", {})
        
        if not synthesis:
            log_pipeline("No synthesis result, using fallback", level="warning")
            synthesis = {
                "report_text": result.get("report", "Error: No report generated"),
                "sources_used": result.get("all_sources", []),
                "citations": [],
                "metadata": {"confidence": 0}
            }
        
        final_result = {
            "query": plan["query"],
            "report_text": synthesis.get("report_text", ""),
            "sources": synthesis.get("sources_used", []),
            "citations": synthesis.get("citations", []),
            "sub_topics": result.get("sub_topics", []),
            "quality_metrics": synthesis.get("metadata", {}),
            "timestamp": datetime.now().isoformat(),
            "iterations": result.get("iteration", 0)
        }
        
        log_pipeline(f"Research complete: {len(final_result['sources'])} sources, {len(final_result['citations'])} cited")
        
        self.last_result = final_result
        
        return final_result
    
    async def execute_research_streaming(
        self,
        plan: Dict,
        on_progress: Optional[Callable[[str], Awaitable]] = None,
        ws = None
    ) -> Dict:
        """Execute research with streaming synthesis."""
        set_current_session(self.session_id)
        
        max_iters = DEPTH_PARAMS.get(plan["depth"], DEPTH_PARAMS["standard"]).get("max_searches", 3)
        
        log_pipeline(f"Executing research with streaming (topics: {len(plan['sub_topics'])})")
        
        init_state: OrchestratorState = {
            "query": plan["query"],
            "depth": plan["depth"],
            "sub_topics": plan["sub_topics"],
            "iteration": 0,
            "max_iterations": max_iters,
            "completed": [],
            "all_sources": [],
            "identified_gaps": [],
            "report": "",
            "synthesis_result": None
        }
        
        if on_progress:
            await on_progress("Researching sub-topics...")
        
        result = await self.graph.ainvoke(init_state)
        
        synthesis = result.get("synthesis_result", {})
        
        if not synthesis:
            log_pipeline("No synthesis result, using fallback", level="warning")
            synthesis = {
                "report_text": result.get("report", "Error: No report generated"),
                "sources_used": result.get("all_sources", []),
                "citations": [],
                "metadata": {"confidence": 0}
            }
        
        report_text = synthesis.get("report_text", "")
        sources = synthesis.get("sources_used", [])
        citations = synthesis.get("citations", [])
        metadata = synthesis.get("metadata", {})
        
        log_pipeline(f"Research complete: {len(sources)} sources, {len(citations)} cited")
        
        if ws:
            await ws.send_json({"type": "synthesis_start"})
            
            chunk_size = 50
            for i in range(0, len(report_text), chunk_size):
                chunk = report_text[i:i+chunk_size]
                await ws.send_json({
                    "type": "synthesis_chunk",
                    "chunk": chunk,
                    "progress": min(100, int((i / len(report_text)) * 100))
                })
                await asyncio.sleep(0.05)
            
            final_result = {
                "query": plan["query"],
                "report_text": report_text,
                "sources": sources,
                "citations": citations,
                "sub_topics": result.get("sub_topics", []),
                "quality_metrics": metadata,
                "timestamp": datetime.now().isoformat(),
                "iterations": result.get("iteration", 0)
            }
            
            await ws.send_json({
                "type": "complete",
                "result": final_result
            })
        else:
            final_result = {
                "query": plan["query"],
                "report_text": report_text,
                "sources": sources,
                "citations": citations,
                "sub_topics": result.get("sub_topics", []),
                "quality_metrics": metadata,
                "timestamp": datetime.now().isoformat(),
                "iterations": result.get("iteration", 0)
            }
        
        log_pipeline(f"Streaming complete")
        
        self.last_result = final_result
        
        return final_result
    
    def clear(self):
        """Clear session-scoped caches."""
        set_current_session(self.session_id)
        
        from src.services import get_cache, get_rag_store
        
        log_pipeline(f"Clearing session {self.session_id[:8]} caches")
        get_cache().clear()
        get_rag_store().clear_all()