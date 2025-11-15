"""Environment-based configuration."""

from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    # Gemini - TWO-TIER STRATEGY (✅ FIXED variable names)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL_FAST: str = os.getenv("GEMINI_MODEL_FAST", "gemini-2.0-flash-exp")
    GEMINI_MODEL_SMART: str = os.getenv("GEMINI_MODEL_SMART", "gemini-2.0-flash")
    
    # Google Search
    GOOGLE_SEARCH_API_KEY: str = os.getenv("GOOGLE_SEARCH_API_KEY", "")
    GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CSE_ID", "")
    
    # Jina
    JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")
    
    @classmethod
    def validate(cls):
        missing = []
        if not cls.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if not cls.GOOGLE_SEARCH_API_KEY:
            missing.append("GOOGLE_SEARCH_API_KEY")
        if not cls.GOOGLE_CSE_ID:
            missing.append("GOOGLE_CSE_ID")
        if not cls.JINA_API_KEY:
            missing.append("JINA_API_KEY")
        
        if missing:
            raise ValueError(f"Missing env vars: {', '.join(missing)}")
        
        print("✓ Configuration loaded")
        print(f"  LLM FAST: {cls.GEMINI_MODEL_FAST}")
        print(f"  LLM SMART: {cls.GEMINI_MODEL_SMART}")


settings = Settings()