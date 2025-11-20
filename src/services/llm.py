"""Gemini LLM wrapper with retry logic."""

import os
import asyncio
import random
import json
from enum import Enum

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from src.utils.logger import log_llm


class LLMTier(Enum):
    """LLM tier selection."""
    FAST = "fast"
    SMART = "smart"


class GeminiLLM:
    """Google Gemini LLM client."""
    
    def __init__(self, model: str, tier_name: str):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model)
        self.tier = tier_name
        self.model_name = model
        
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    async def generate(self, prompt: str, max_tokens: int = 4000, json_mode: bool = False) -> str:
        """Generate text content with retry on rate limits"""
        generation_config = {"max_output_tokens": max_tokens}
        if json_mode:
            generation_config["response_mime_type"] = "application/json"
        
        for attempt in range(5):
            try:
                response = await self.model.generate_content_async(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=self.safety_settings
                )
                
                if response.parts:
                    return response.text
                
                if response.candidates:
                    finish_reason = response.candidates[0].finish_reason
                    if finish_reason == 2:
                        return response.text
                    if finish_reason in [3, 4]:
                        return '{}' if json_mode else "Content blocked by safety filters."
                
                return '{}' if json_mode else ""
                
            except Exception as e:
                if not self._should_retry(e, attempt):
                    log_llm(f"Error: {str(e)[:100]}", level="error", tier=self.tier)
                    raise e
                
                delay = self._calculate_delay(attempt)
                log_llm(f"Rate limit. Retrying in {delay:.1f}s...", level="warning", tier=self.tier)
                await asyncio.sleep(delay)
                
        raise Exception("Max retries exceeded")
    
    async def generate_json(self, prompt: str, max_tokens: int = 4000) -> dict:
        """Generate JSON output."""
        json_str = await self.generate(prompt, max_tokens=max_tokens, json_mode=True)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}
    
    def _should_retry(self, error: Exception, attempt: int) -> bool:
        """Check if error is retryable."""
        if attempt >= 4:
            return False
        error_str = str(error)
        return "429" in error_str or "Resource exhausted" in error_str
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        jitter = random.uniform(0.1, 1.5)
        return (2.0 * (2 ** attempt)) + jitter