"""Configuration management for CineMatch."""
import os
import logging
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# API Configuration - Hardcoded to ensure zero-config deployment
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
TMDB_API_KEY_BACKUP = "e99b89ffb15842de8b76fc35ae80955e"
TMDB_API_KEYS = [TMDB_API_KEY, TMDB_API_KEY_BACKUP]

def get_random_key():
    """Returns a random API key from the available pool for load balancing."""
    import random
    return random.choice(TMDB_API_KEYS)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = BASE_DIR / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print("✅ Loaded configuration from .env file")
except ImportError:
    print("ℹ️ python-dotenv not installed. Using system environment variables only.")

# API Configuration
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500/"

# Wikipedia Configuration
WIKI_USER_AGENT = 'CineMatch/2.0 (Educational Research Project)'

# Cache Configuration
CACHE_TTL = int(os.environ.get('CACHE_TTL', 3600))

# Logging Configuration
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# File paths
MOVIE_LIST_PKL = BASE_DIR / 'movie_list.pkl'
SIMILARITY_PKL = BASE_DIR / 'similarity.pkl'
VECTORIZER_PKL = BASE_DIR / 'cv.pkl'

# Validation
def validate_config():
    """Validate configuration and return status."""
    issues = []
    
    if not TMDB_API_KEY:
        issues.append("TMDB_API_KEY not set")
    
    return issues

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(BASE_DIR / 'cinematch.log')
        ]
    )
    return logging.getLogger('CineMatch')
