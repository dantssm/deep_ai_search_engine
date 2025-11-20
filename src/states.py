"""States with quality metrics and structured results"""

from typing import TypedDict, List, Optional, Dict


class RAGContext(TypedDict):
    """Metadata about the vector store collection for a specific topic"""
    collection_id: str
    chunks_indexed: int


class QualityMetrics(TypedDict):
    """Metrics to evaluate the success of a research task"""
    confidence: float
    source_count: int
    iterations_used: int
    chunks_retrieved: int


class RetryInfo(TypedDict):
    """Metadata for tracking failed attempts at a topic"""
    topic: str
    attempt: int
    last_error: str


class ResearcherState(TypedDict):
    """State for individual researcher agent."""
    topic: str
    parent_query: str
    
    searches: List[str]
    scraped_urls: List[str]

    rag: RAGContext
    scraped_content: List[dict]
    retrieved_chunks: List[dict]

    reflections: List[Dict]
    iteration: int
    max_iterations: int
    
    findings: str
    sources: List[dict]
    gaps: List[str]
    quality_metrics: Optional[QualityMetrics]


class OrchestratorState(TypedDict):
    """The global memory of the Orchestrator. Scope: The entire user session"""
    query: str
    depth: str
    sub_topics: List[str]
    iteration: int
    max_iterations: int
    completed: List[dict]
    all_sources: List[dict]
    identified_gaps: List[str]
    report: str
    retry_queue: List[RetryInfo]
    overall_quality: Optional[QualityMetrics]
    synthesis_result: Optional[Dict]


class SynthesisResult(TypedDict):
    """Structured synthesis output."""
    report_text: str
    sources_used: List[Dict]
    citations: List[int]
    metadata: Dict


class ResearchResult(TypedDict):
    """Final research output."""
    query: str
    report_text: str
    sources: List[Dict]
    citations: List[int]
    sub_topics: List[str]
    quality_metrics: Dict
    timestamp: str
    iterations: int