import pytest

from app.browser_executor import ActionResult
from app.schemas import TestCase


@pytest.fixture
def sample_test_case() -> TestCase:
    return TestCase(
        title="User can log in with valid credentials",
        start_url="https://example.com/login",
        steps=[
            "Enter 'demo_user' into the username field",
            "Click the login button",
        ],
        expected_result="The dashboard page is shown with a welcome message",
    )


class FakeBrowserExecutor:
    """Drop-in replacement for BrowserExecutor that never launches Chromium."""

    def __init__(self, headless: bool = True, page_text: str = "Welcome, demo_user!"):
        self.headless = headless
        self._page_text = page_text
        self.started_with: str | None = None
        self.stopped = False
        self.calls: list[tuple[str, tuple]] = []

    async def start(self, start_url: str) -> None:
        self.started_with = start_url

    async def stop(self) -> None:
        self.stopped = True

    async def click(self, selector: str) -> ActionResult:
        self.calls.append(("click", (selector,)))
        return ActionResult(True, f"clicked {selector}")

    async def fill(self, selector: str, text: str) -> ActionResult:
        self.calls.append(("fill", (selector, text)))
        return ActionResult(True, f"filled {selector}")

    async def get_visible_text(self) -> str:
        return self._page_text

    async def current_url(self) -> str:
        return self.started_with or ""


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Returns pre-scripted JSON responses in order, one per .ainvoke() call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = 0

    async def ainvoke(self, messages):
        response = self._responses[self.calls]
        self.calls += 1
        return FakeLLMResponse(response)


@pytest.fixture
def fake_browser() -> FakeBrowserExecutor:
    return FakeBrowserExecutor()
