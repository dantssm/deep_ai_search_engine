# src/config/__init__.py
"""Configuration package."""

from .settings import settings
from .constants import DEPTH_PARAMS

config = settings

__all__ = ["settings", "config", "DEPTH_PARAMS"]