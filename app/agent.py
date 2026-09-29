"""LangGraph state machine that turns a TestCase into executed browser
actions and a pass/fail verdict.

Graph shape:

    plan_step -> execute_step -> [more steps?] -> plan_step (loop)
                                -> [no more steps / error] -> verify -> END
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from app.browser_executor import ActionResult, BrowserExecutor
from app.config import get_settings
from app.schemas import Plan, StepResult, TestCase

PLANNER_SYSTEM_PROMPT = """You are a QA automation planner.

Given:
- A single natural language test step
- The visible text of the current web page

Return ONLY valid JSON:

{
  "action": "click" | "fill",
  "selector": "<Playwright-compatible selector>",
  "text": "<text to type, only for fill>"
}

STRICT RULES:
- Prefer Playwright text selectors when possible:
  - text=Submit
  - text="More information"
- For inputs:
  - input[name="username"]
  - input[placeholder="Email"]
- For buttons:
  - button:has-text("Login")
- NEVER use invalid CSS like :contains
- Selector must be usable in Playwright directly
- Keep selectors simple and robust

If unsure, prefer text-based selectors.

Return ONLY JSON. No explanation.
"""

VERIFIER_SYSTEM_PROMPT = """You are a QA automation verifier. Given the \
expected result of a test case and the final page's visible text, respond \
with ONLY a JSON object:

{"passed": true|false, "reason": "<one sentence explaining why>"}

No prose, only JSON."""

# Total wall-clock budget for one test run (all steps + verify). With
# max_retries=0 and a 12s per-call LLM timeout, one call fails fast rather
# than silently retrying — this ceiling exists to bound the browser-side
# fallback-selector hunting below, not to accommodate retries.
RUN_TIMEOUT_SECONDS = 120

# Hard ceiling on steps per run, independent of how many the test case has,
# as a defensive backstop against a runaway loop.
MAX_STEPS = 10

# Kept short deliberately: each entry costs one real browser_timeout_ms wait
# when the primary selector fails, so a long list here is what actually made
# a single failed step slow, not the LLM call.
CLICK_FALLBACK_SELECTORS = [
    "text=Login",
    "button",
]

FILL_FALLBACK_SELECTORS = [
    "input[name='username']",
    "input[type='text']",
]


class AgentState(TypedDict):
    test_case: TestCase
    step_index: int
    step_results: list[StepResult]
    passed: Optional[bool]
    verdict_reason: str
    error: Optional[str]
    pending_plan: dict
    step_started_at: float
    llm_calls: int
    input_tokens: int
    output_tokens: int


def _get_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0,
        # Our own try/except around every ainvoke() call already turns a
        # failure into a clean, fast "failed" result — so the client doesn't
        # need to silently retry on top of that. A retry here (previously
        # max_retries=2, timeout=25) could legitimately take up to ~75s for
        # ONE call alone, which blew straight through RUN_TIMEOUT_SECONDS
        # before a single call even finished. Fail fast instead; you can
        # just rerun the test if it was a one-off hiccup.
        max_retries=0,
        timeout=12,
    )


def _response_text(response) -> str:
    """Normalize response.content, which some Gemini models return as a
    plain string and others return as a list of content parts (e.g. when
    "thinking" output accompanies the actual answer)."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return str(content)


def _parse_json_response(raw: str) -> dict:
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


def _sanitize_selector(selector: str) -> str:
    if not selector:
        return "body"
    if ":contains" in selector:
        selector = selector.replace(":contains", "")
    if len(selector) > 200:
        return "body"
    return selector.strip()


def _elapsed_ms(state: AgentState) -> int:
    return int((time.perf_counter() - state["step_started_at"]) * 1000)


def _record_usage(state: AgentState, response) -> None:
    """Count one LLM call and add its token usage, when the response reports it."""
    state["llm_calls"] += 1
    usage = getattr(response, "usage_metadata", None) or {}
    state["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
    state["output_tokens"] += int(usage.get("output_tokens", 0) or 0)


class QAAgent:
    """Wraps the compiled LangGraph graph plus the browser session it drives."""

    def __init__(self, browser: BrowserExecutor, llm: Optional[ChatGoogleGenerativeAI] = None):
        self.browser = browser
        self.llm = llm or _get_llm()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("plan_step", self._plan_step)
        graph.add_node("execute_step", self._execute_step)
        graph.add_node("verify", self._verify)

        graph.set_entry_point("plan_step")
        graph.add_edge("plan_step", "execute_step")
        graph.add_conditional_edges(
            "execute_step",
            self._route_after_execute,
            {"continue": "plan_step", "verify": "verify"},
        )
        graph.add_edge("verify", END)
        return graph.compile()

    async def _plan_step(self, state: AgentState) -> AgentState:
        state["step_started_at"] = time.perf_counter()
        step_text = state["test_case"].steps[state["step_index"]]
        page_text = await self.browser.get_visible_text()
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Test step:\n{step_text}\n\nVisible page text:\n{page_text[:2000]}"
            ),
        ]
        try:
            response = await self.llm.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001 - any LLM/network failure (timeout, 429, 5xx, etc.)
            # Treat a failed call the same way as a malformed response: no
            # action to take this step, fail gracefully instead of letting
            # an arbitrary exception type escape the graph uncaught.
            # str(exc) can be empty for some exception types (certain client-
            # side timeouts carry no message) — fall back to the exception's
            # class name so the verdict is never blank.
            detail = str(exc) or exc.__class__.__name__
            state["error"] = f"planner call failed: {detail}"
            state["pending_plan"] = {"action": "invalid", "selector": "", "text": ""}
            return state

        _record_usage(state, response)
        try:
            plan = Plan.model_validate(_parse_json_response(_response_text(response))).model_dump()
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
            state["error"] = f"planner returned invalid JSON: {exc}"
            plan = {"action": "invalid", "selector": "", "text": ""}
        state["pending_plan"] = plan
        return state

    async def _execute_step(self, state: AgentState) -> AgentState:
        plan = state.get("pending_plan", {})
        step_text = state["test_case"].steps[state["step_index"]]
        action = plan.get("action")

        if action not in ("click", "fill"):
            # Planner failed to produce a usable action (invalid JSON, or an
            # action outside our click/fill vocabulary). Nothing was
            # attempted against the browser — say so plainly rather than
            # showing a leftover default selector like "body", which would
            # wrongly imply an action was tried.
            reason = state.get("error") or "planner action was rejected"
            state["step_results"].append(
                StepResult(
                    step=step_text,
                    action_taken=f"{action} -> [rejected]",
                    success=False,
                    detail=reason,
                    duration_ms=_elapsed_ms(state),
                )
            )
            state["error"] = reason
            state["step_index"] += 1
            return state

        selector = _sanitize_selector(plan.get("selector", "body"))
        print(f"STEP: {step_text} | PLAN: {plan} | SELECTOR: {selector}")

        if action == "fill":
            text = plan.get("text", "")
            result = await self.browser.fill(selector, text)
            if not result.success:
                for fallback in FILL_FALLBACK_SELECTORS:
                    result = await self.browser.fill(fallback, text)
                    if result.success:
                        selector = fallback
                        break
        else:  # action == "click"
            result = await self.browser.click(selector)
            if not result.success:
                for fallback in CLICK_FALLBACK_SELECTORS:
                    result = await self.browser.click(fallback)
                    if result.success:
                        selector = fallback
                        break

        state["step_results"].append(
            StepResult(
                step=step_text,
                action_taken=f"{action} -> {selector}",
                success=result.success,
                detail=result.detail,
                duration_ms=_elapsed_ms(state),
            )
        )
        if not result.success:
            state["error"] = result.detail
        state["step_index"] += 1
        return state

    def _route_after_execute(self, state: AgentState) -> Literal["continue", "verify"]:
        if state.get("error"):
            return "verify"
        if state["step_index"] >= MAX_STEPS:
            state["error"] = "maximum number of steps reached"
            return "verify"
        if state["step_index"] < len(state["test_case"].steps):
            return "continue"
        return "verify"

    async def _verify(self, state: AgentState) -> AgentState:
        if state.get("error"):
            state["passed"] = False
            state["verdict_reason"] = state["error"]
            return state

        page_text = await self.browser.get_visible_text()
        messages = [
            SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Expected result: {state['test_case'].expected_result}\n\n"
                    f"Final page text:\n{page_text}"
                )
            ),
        ]
        try:
            response = await self.llm.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001 - any LLM/network failure (timeout, 429, 5xx, etc.)
            state["passed"] = False
            detail = str(exc) or exc.__class__.__name__
            state["verdict_reason"] = f"verifier call failed: {detail}"
            return state

        _record_usage(state, response)
        try:
            verdict = _parse_json_response(_response_text(response))
            state["passed"] = bool(verdict.get("passed", False))
            state["verdict_reason"] = verdict.get("reason", "")
        except (json.JSONDecodeError, AttributeError) as exc:
            state["passed"] = False
            state["verdict_reason"] = f"verifier returned invalid JSON: {exc}"
        return state

    async def run(self, test_case: TestCase) -> AgentState:
        await self.browser.start(str(test_case.start_url))
        try:
            initial_state: AgentState = {
                "test_case": test_case,
                "step_index": 0,
                "step_results": [],
                "passed": None,
                "verdict_reason": "",
                "error": None,
                "pending_plan": {},
                "step_started_at": 0.0,
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            final_state = await asyncio.wait_for(
                self.graph.ainvoke(initial_state), timeout=RUN_TIMEOUT_SECONDS
            )
            return final_state
        except asyncio.TimeoutError:
            return {
                "test_case": test_case,
                "step_index": 0,
                "step_results": [],
                "passed": False,
                "verdict_reason": f"test run timed out after {RUN_TIMEOUT_SECONDS}s",
                "error": f"timeout after {RUN_TIMEOUT_SECONDS} seconds",
                "pending_plan": {},
            }
        finally:
            await self.browser.stop()