import os
from functools import lru_cache

from dotenv import load_dotenv

# Loads .env into the process environment. Harmless if .env doesn't exist
# (e.g. inside Docker, where env vars are injected by docker-compose instead) —
# load_dotenv() just silently does nothing in that case.
load_dotenv()


class Settings:
    def __init__(self) -> None:
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        self.headless_browser = os.getenv("HEADLESS_BROWSER", "true").lower() != "false"
        self.api_key = os.getenv("API_KEY", "").strip()
        self.allowed_hosts = [
            h.strip()
            for h in os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1,testserver").split(",")
            if h.strip()
        ]
        self.allowed_url_hosts = frozenset(
            h.strip().lower().rstrip(".")
            for h in os.getenv("ALLOWED_URL_HOSTS", "").split(",")
            if h.strip()
        )
        self.max_concurrent_runs = _positive_int("MAX_CONCURRENT_RUNS", 2)
        self.max_stored_runs = _positive_int("MAX_STORED_RUNS", 1000)
        self.browser_timeout_ms = _positive_int("BROWSER_TIMEOUT_MS", 15000)
        self.screenshots_enabled = os.getenv("SCREENSHOTS_ENABLED", "false").lower() == "true"


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


@lru_cache
def get_settings() -> Settings:
    return Settings()
