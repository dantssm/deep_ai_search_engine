import os
import asyncio
import random
import json
from typing import Dict
from enum import Enum

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from src.utils.logger import log_llm

class LLMTier(Enum):
    FAST = "fast"
    SMART = "smart"

class GeminiLLM:
    """Wrapper for Google Gemini LLM"""
    
    def __init__(self, model: str, tier_name: str):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model)
        self.tier = tier_name
        self.model_name = model
    
    async def generate(self, prompt: str, max_tokens: int = 4000, json_mode: bool = False) -> str:
        """Generate text using the Gemini LLM"""
        max_retries = 5 
        base_delay = 2.0
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        generation_config = {"max_output_tokens": max_tokens}
        
        if json_mode:
            generation_config["response_mime_type"] = "application/json"
        
        for attempt in range(max_retries):
            try:
                response = await self.model.generate_content_async(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                if response.parts:
                    return response.text
                
                if response.candidates:
                    finish_reason = response.candidates[0].finish_reason
                    if finish_reason == 2:
                        return response.text
                    elif finish_reason in [3, 4]:
                        if json_mode: return '{}'
                        return "Content blocked by safety filters."
                
                return '{}' if json_mode else ""
                
            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "Resource exhausted" in error_str
                
                if is_rate_limit and attempt < max_retries - 1:
                    jitter = random.uniform(0.1, 1.5)
                    delay = (base_delay * (2 ** attempt)) + jitter
                    
                    log_llm(f"Rate limit. Retrying in {delay:.1f}s...", level="warning", tier=self.tier)
                    
                    await asyncio.sleep(delay)
                    continue
                
                log_llm(f"Error: {error_str[:100]}", level="error", tier=self.tier)
                raise e
                
        raise Exception("Max retries exceeded")
    
    async def generate_json(self, prompt: str, max_tokens: int = 4000) -> Dict:
        """Generate JSON output using the Gemini LLM"""
        json_str = await self.generate(prompt, max_tokens=max_tokens, json_mode=True)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}