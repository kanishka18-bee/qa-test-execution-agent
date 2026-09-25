"""LangGraph state machine that turns a TestCase into executed browser
actions and a pass/fail verdict.

Graph shape:

    plan_step -> execute_step -> [more steps?] -> plan_step (loop)
                                -> [no more steps / error] -> verify -> END

`plan_step` asks the LLM to translate one natural-language test step into a
single concrete browser action (click/fill) plus a CSS selector, grounded in
the current page's visible text. `verify` asks the LLM to compare the final
page state against the test case's expected_result.
"""
from __future__ import annotations

import json
from typing import Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from app.browser_executor import ActionResult, BrowserExecutor
from app.config import get_settings
from app.schemas import Plan, StepResult, TestCase

PLANNER_SYSTEM_PROMPT = """You are a QA automation planner. Given a single \
natural-language test step and the visible text of the current web page, \
respond with ONLY a JSON object describing one browser action to take:

{"action": "click"|"fill", "selector": "<a plausible CSS selector>", \
"text": "<text to type, only for fill>"}

Infer a reasonable CSS selector from the page text and common conventions \
(e.g. input[name='username'], button[type='submit']). No prose, only JSON."""

VERIFIER_SYSTEM_PROMPT = """You are a QA automation verifier. Given the \
expected result of a test case and the final page's visible text, respond \
with ONLY a JSON object:

{"passed": true|false, "reason": "<one sentence explaining why>"}

No prose, only JSON."""


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
    )


def _parse_json_response(raw: str) -> dict:
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


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
            HumanMessage(content=f"Step: {step_text}\n\nPage text:\n{page_text}"),
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
        selector = plan.get("selector", "body")

        if state.get("error"):
            result = ActionResult(False, state["error"])
        elif action == "fill":
            result = await self.browser.fill(selector, plan.get("text", ""))
        elif action == "click":
            result = await self.browser.click(selector)
        else:
            result = ActionResult(False, "planner action was rejected")

        state["step_results"].append(
            StepResult(
                step=step_text,
                action_taken=f"{action} -> {selector or '[rejected]'}",
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
            final_state = await self.graph.ainvoke(initial_state)
            return final_state
        finally:
            await self.browser.stop()
