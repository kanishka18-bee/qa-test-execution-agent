import os
from functools import lru_cache


class Settings:
    gemini_api_key: str
    gemini_model: str
    headless_browser: bool
    api_key: str
    allowed_hosts: list[str]
    allowed_url_hosts: frozenset[str]
    max_concurrent_runs: int
    max_stored_runs: int
    browser_timeout_ms: int
    screenshots_enabled: bool

    def __init__(self) -> None:
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
        self.headless_browser = os.getenv("HEADLESS_BROWSER", "true").lower() != "false"
        self.api_key = os.getenv("API_KEY", "").strip()
        self.allowed_hosts = [
            host.strip()
            for host in os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1,testserver").split(",")
            if host.strip()
        ]
        self.allowed_url_hosts = frozenset(
            host.strip().lower().rstrip(".")
            for host in os.getenv("ALLOWED_URL_HOSTS", "").split(",")
            if host.strip()
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
