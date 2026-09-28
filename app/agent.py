"""LangGraph state machine that turns a TestCase into executed browser
actions and a pass/fail verdict.

Graph shape:

    plan_step -> execute_step -> [more steps?] -> plan_step (loop)
                                -> [no more steps / error] -> verify -> END
"""
from __future__ import annotations

import asyncio
import json
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

# Total wall-clock budget for one test run (all steps + verify). Kept
# separate from BROWSER_TIMEOUT_MS (the per-selector wait in
# browser_executor) since this caps the whole graph, not one action.
RUN_TIMEOUT_SECONDS = 60

# Hard ceiling on steps per run, independent of how many the test case has,
# as a defensive backstop against a runaway loop.
MAX_STEPS = 10

CLICK_FALLBACK_SELECTORS = [
    "text=Login",
    "text=Submit",
    "text=Sign in",
    "button",
]

FILL_FALLBACK_SELECTORS = [
    "input[name='username']",
    "input[name='email']",
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


def _get_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0,
        max_retries=1,
        timeout=20,
    )


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
        step_text = state["test_case"].steps[state["step_index"]]
        page_text = await self.browser.get_visible_text()
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Test step:\n{step_text}\n\nVisible page text:\n{page_text[:2000]}"
            ),
        ]
        response = await self.llm.ainvoke(messages)
        try:
            plan = Plan.model_validate(_parse_json_response(response.content)).model_dump()
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
        response = await self.llm.ainvoke(messages)
        try:
            verdict = _parse_json_response(response.content)
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
