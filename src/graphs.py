"""LangGraph definitions for orchestrator and researcher."""

from langgraph.graph import StateGraph, START, END
from src.states import ResearcherState, OrchestratorState
from src.nodes.researcher import (
    search_node, scrape_and_index_node, retrieve_node,
    reflect_node, summarize_node, should_continue
)
from src.nodes.orchestrator import (
    plan_node, dispatch_node, critique_node,
    synthesize_node, should_continue_orchestrator
)


def build_researcher_graph():
    """
    Researcher graph: Investigates a single topic.
    
    Flow: search → scrape+index → retrieve → reflect
            ↑_____________________________________|
            (loops until max iterations or done)
    """
    g = StateGraph(ResearcherState)
    
    g.add_node("search", search_node)
    g.add_node("scrape_index", scrape_and_index_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("reflect", reflect_node)
    g.add_node("summarize", summarize_node)
    
    g.add_edge(START, "search")
    g.add_edge("search", "scrape_index")
    g.add_edge("scrape_index", "retrieve")
    g.add_edge("retrieve", "reflect")
    
    g.add_conditional_edges("reflect", should_continue, {
        "search": "search",
        "summarize": "summarize"
    })
    
    g.add_edge("summarize", END)
    
    return g.compile()


def build_orchestrator_graph():
    """
    Orchestrator graph: Coordinates multiple researchers.
    
    Flow: plan → dispatch → critique → synthesize
                    ↑_________|
                    (loops if needed)
    """
    g = StateGraph(OrchestratorState)
    
    g.add_node("plan", plan_node)
    g.add_node("dispatch", dispatch_node)
    g.add_node("critique", critique_node)
    g.add_node("synthesize", synthesize_node)
    
    g.add_edge(START, "plan")
    g.add_edge("plan", "dispatch")
    g.add_edge("dispatch", "critique")
 
    g.add_conditional_edges("critique", should_continue_orchestrator, {
        "dispatch": "dispatch",
        "synthesize": "synthesize"
    })
    
    g.add_edge("synthesize", END)
    
    return g.compile()