"""Centralized logging configuration with WebSocket broadcast support."""

import logging
import sys
import asyncio
import threading
from typing import Optional, Callable, List

_log_callbacks: List[Callable[[str], None]] = []
_callbacks_lock = threading.Lock()

def add_log_callback(callback: Callable[[str], None]):
    """Register a callback to receive log messages (thread-safe)."""
    with _callbacks_lock:
        _log_callbacks.append(callback)

def remove_log_callback(callback: Callable[[str], None]):
    """Remove a registered callback (thread-safe)."""
    with _callbacks_lock:
        if callback in _log_callbacks:
            _log_callbacks.remove(callback)

def broadcast_log(msg: str):
    """Send log message to all registered callbacks (thread-safe)."""
    with _callbacks_lock:
        callbacks_copy = _log_callbacks.copy()
    
    for callback in callbacks_copy:
        try:
            if asyncio.iscoroutinefunction(callback):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(callback(msg))
                except RuntimeError:
                    pass
            else:
                callback(msg)
        except Exception:
            pass

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output."""
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logger(name: str = "research", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = ColoredFormatter(
        fmt='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

search_logger = setup_logger("research.search")
scrape_logger = setup_logger("research.scrape")
rag_logger = setup_logger("research.rag")
llm_logger = setup_logger("research.llm")
pipeline_logger = setup_logger("research.pipeline")
orchestrator_logger = setup_logger("research.orchestrator")
researcher_logger = setup_logger("research.researcher")

def log_search(msg: str, level: str = "info"):
    getattr(search_logger, level.lower())(msg)
    broadcast_log(f"[Search] {msg}")

def log_scrape(msg: str, level: str = "info"):
    getattr(scrape_logger, level.lower())(msg)
    broadcast_log(f"[Scrape] {msg}")

def log_rag(msg: str, level: str = "info"):
    getattr(rag_logger, level.lower())(msg)
    broadcast_log(f"[RAG] {msg}")

def log_llm(msg: str, level: str = "info", tier: Optional[str] = None):
    if tier: msg = f"[{tier}] {msg}"
    getattr(llm_logger, level.lower())(msg)

def log_pipeline(msg: str, level: str = "info"):
    getattr(pipeline_logger, level.lower())(msg)
    broadcast_log(f"[System] {msg}")

def log_orchestrator(msg: str, level: str = "info"):
    getattr(orchestrator_logger, level.lower())(msg)
    broadcast_log(f"{msg}")

def log_researcher(msg: str, level: str = "info"):
    getattr(researcher_logger, level.lower())(msg)
    broadcast_log(f"  {msg}")