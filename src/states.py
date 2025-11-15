# src/states.py
"""Enhanced states with quality metrics and structured results."""

from typing import TypedDict, List, Optional, Dict

class RAGContext(TypedDict):
    collection_id: str
    chunks_indexed: int

class QualityMetrics(TypedDict):
    confidence: float  # 0-1 score
    source_count: int
    source_diversity: float
    gap_count: int
    fact_count: int
    coverage_score: float

class RetryInfo(TypedDict):
    topic: str
    attempt: int
    last_error: str

class ResearcherState(TypedDict):
    topic: str
    parent_query: str
    
    searches: List[str]
    scraped_urls: List[str]
    rag: RAGContext
    reflections: List[Dict]
    iteration: int
    max_iterations: int
    
    scraped_content: List[dict]
    retrieved_chunks: List[dict]
    
    findings: str
    sources: List[dict]
    gaps: List[str]
    quality_metrics: Optional[QualityMetrics]

class OrchestratorState(TypedDict):
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