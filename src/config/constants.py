"""Global Constants"""

DEPTH_PARAMS = {
    "standard": {
        "max_results": 12, 
        "sources_range": "10-15", 
        "max_searches": 5,
        "researcher_iterations": 2
    },
    "deep": {
        "max_results": 18, 
        "sources_range": "15-25", 
        "max_searches": 8,
        "researcher_iterations": 3
    }
}

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 50

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v2-base-en"

JINA_SCRAPE_URL = "https://r.jina.ai/"
MAX_CONTENT_LENGTH = 6000
SCRAPE_TIMEOUT = 15.0

GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
DEFAULT_SEARCH_RESULTS = 10

MAX_ANSWER_TOKENS = 10000