"""Gemini LLM wrapper with retry logic and tier management"""

import os
import asyncio
import random
import json
from enum import Enum
from typing import Dict

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from src.utils.logger import log_llm


class LLMTier(Enum):
    """    
    FAST: Used for quick tasks like reflection, simple summaries.
    SMART: Used for complex planning, deep reasoning, and final synthesis.
    """
    FAST = "fast"
    SMART = "smart"


class GeminiLLM:
    """Wrapper for Google's Gemini models"""
    
    def __init__(self, model: str, tier_name: str):
        """
        Initialize the Gemini wrapper.

        Args:
            model (str): The model identifier.
            tier_name (str): 'FAST' or 'SMART' for logging context.
        """
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model)
        self.tier = tier_name
        self.model_name = model
        
        # Disable safety filters
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    async def generate(self, prompt: str, max_tokens: int = 4000, json_mode: bool = False) -> str:
        """
        Generate text from the LLM.

        Args:
            prompt (str): The input prompt.
            max_tokens (int): Max output tokens.
            json_mode (bool): If True, requests JSON MIME type.

        Returns:
            str: The generated text. If json_mode is True and generation fails, returns '{}'.

        Raises:
            Exception: If max retries are exceeded or a non-retryable error occurs.
        """
        generation_config = {"max_output_tokens": max_tokens}
        if json_mode:
            generation_config["response_mime_type"] = "application/json"
        
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                response = await self.model.generate_content_async(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=self.safety_settings
                )
                
                # 99% of the time, valid content is in parts
                if response.parts:
                    return response.text
                
                # Fallback checks
                if response.candidates:
                    finish_reason = response.candidates[0].finish_reason
                    if finish_reason == 2:
                        return response.text
                    # 3 (SAFETY) or 4 (RECITATION)
                    if finish_reason in [3, 4]:
                        log_llm("Content blocked by safety filters", level="warning", tier=self.tier)
                        return '{}' if json_mode else "Content blocked."
                
                return '{}' if json_mode else ""
                
            except Exception as e:
                if not self._should_retry(e, attempt, max_retries):
                    log_llm(f"Error: {str(e)[:100]}", level="error", tier=self.tier)
                    raise e
                
                delay = self._calculate_delay(attempt)
                log_llm(f"Rate limit. Retrying in {delay:.1f}s...", level="warning", tier=self.tier)
                await asyncio.sleep(delay)
                
        raise Exception("Max retries exceeded for LLM generation")
    
    async def generate_json(self, prompt: str, max_tokens: int = 4000) -> Dict:
        """
        Helper method to generate and parse JSON directly.

        Args:
            prompt (str): Input prompt.
            max_tokens (int): Max tokens.

        Returns:
            Dict: Parsed JSON dictionary.
        """
        json_str = await self.generate(prompt, max_tokens=max_tokens, json_mode=True)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            log_llm("Failed to decode JSON response", level="error", tier=self.tier)
            return {}
    
    def _should_retry(self, error: Exception, attempt: int, max_retries: int) -> bool:
        """Determine if the error is retryable (Rate Limits)"""
        if attempt >= max_retries - 1:
            return False
            
        error_str = str(error)
        return "429" in error_str or "Resource exhausted" in error_str
    
    def _calculate_delay(self, attempt: int) -> float:
        """Formula: 2 * (2^attempt) + random_jitter"""
        jitter = random.uniform(0.1, 1.5)
        return (2.0 * (2 ** attempt)) + jitter