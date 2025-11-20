"""Session management and service isolation."""

import os
import asyncio
import threading
from typing import Optional, Dict

from src.cache.memory_cache import SimpleCache, CachedGoogleSearcher, CachedJinaScraper
from src.rag.store import RAGStore
from .llm import GeminiLLM, LLMTier

class SessionServices:
    """Isolated services for a single user session"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = asyncio.get_event_loop().time()
        
        self._cache = SimpleCache()
        self._searcher = CachedGoogleSearcher(
            os.getenv("GOOGLE_SEARCH_API_KEY"),
            os.getenv("GOOGLE_CSE_ID"),
            self._cache
        )
        self._scraper = CachedJinaScraper(self._cache)
        self._rag = RAGStore(os.getenv("JINA_API_KEY"))
        
        self._llm_fast = None
        self._llm_smart = None
    
    def get_cache(self) -> SimpleCache:
        return self._cache
    
    def get_searcher(self) -> CachedGoogleSearcher:
        return self._searcher
    
    def get_scraper(self) -> CachedJinaScraper:
        return self._scraper
    
    def get_rag_store(self) -> RAGStore:
        return self._rag
    
    def get_llm(self, tier: LLMTier = LLMTier.FAST):
        if tier == LLMTier.FAST:
            if self._llm_fast is None:
                self._llm_fast = GeminiLLM(os.getenv("GEMINI_MODEL_FAST", "gemini-2.0-flash-lite"), "FAST")
            return self._llm_fast
        else:
            if self._llm_smart is None:
                self._llm_smart = GeminiLLM(os.getenv("GEMINI_MODEL_SMART", "gemini-2.0-flash"), "SMART")
            return self._llm_smart
    
    def cleanup(self):
        print(f"Cleaning up session {self.session_id[:8]}...")
        if self._cache: self._cache.clear()
        if self._rag: self._rag.clear_all()
        print(f"Session {self.session_id[:8]} cleaned")

class SessionManager:
    """Global manager that holds all active user sessions"""
    
    def __init__(self):
        self._sessions: Dict[str, SessionServices] = {}
        self._lock = threading.Lock()
    
    def get_or_create_session(self, session_id: str) -> SessionServices:
        """Retrieve existing session or start a new one"""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionServices(session_id)
                print(f"Created new session: {session_id[:8]}")
            return self._sessions[session_id]
    
    def cleanup_session(self, session_id: str):
        """Delete a session and free memory"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].cleanup()
                del self._sessions[session_id]
    
    def get_active_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)
    
    def cleanup_old_sessions(self, max_age_seconds: float = 3600):
        """Garbage collection for abandoned sessions"""
        with self._lock:
            now = asyncio.get_event_loop().time()
            old_sessions = [
                sid for sid, session in self._sessions.items()
                if now - session.created_at > max_age_seconds
            ]
            for sid in old_sessions:
                self._sessions[sid].cleanup()
                del self._sessions[sid]

_session_manager = SessionManager()
_current_session_id = threading.local()

def get_session_manager() -> SessionManager:
    return _session_manager

def set_current_session(session_id: str):
    _current_session_id.value = session_id

def get_current_session() -> Optional[str]:
    return getattr(_current_session_id, 'value', None)

def get_current_services() -> SessionServices:
    session_id = get_current_session()
    if not session_id:
        raise RuntimeError("No session set for current context")
    return _session_manager.get_or_create_session(session_id)

def get_memory_stats() -> dict:
    """Get system memory usage statistics"""
    try:
        import psutil
        import os as os_module
        process = psutil.Process(os_module.getpid())
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "percent": process.memory_percent(),
            "active_sessions": _session_manager.get_active_sessions(),
            "available": True
        }
    except ImportError:
        return {
            "available": False, 
            "active_sessions": _session_manager.get_active_sessions()
        }