"""Services package exposing LLM and Session utilities"""

from .llm import LLMTier, GeminiLLM
from .session_manager import get_session_manager, set_current_session, get_current_session, get_current_services, get_memory_stats

def get_cache():
    return get_current_services().get_cache()

def get_searcher():
    return get_current_services().get_searcher()

def get_scraper():
    return get_current_services().get_scraper()

def get_rag_store():
    return get_current_services().get_rag_store()

def get_llm(tier: LLMTier = LLMTier.FAST):
    return get_current_services().get_llm(tier)

__all__ = [
    "LLMTier", 
    "GeminiLLM",
    "get_session_manager",
    "set_current_session",
    "get_current_session",
    "get_memory_stats",
    "get_cache",
    "get_searcher",
    "get_scraper",
    "get_rag_store",
    "get_llm"
]