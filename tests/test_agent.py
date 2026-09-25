import pytest

from app.agent import QAAgent
from tests.conftest import FakeLLM


@pytest.mark.asyncio
async def test_agent_passes_when_all_steps_succeed_and_result_matches(
    sample_test_case, fake_browser
):
    llm = FakeLLM(
        responses=[
            '{"action": "fill", "selector": "input[name=\'username\']", "text": "demo_user"}',
            '{"action": "click", "selector": "button[type=\'submit\']"}',
            '{"passed": true, "reason": "Welcome message for demo_user is visible"}',
        ]
    )
    agent = QAAgent(browser=fake_browser, llm=llm)

    final_state = await agent.run(sample_test_case)

    assert final_state["passed"] is True
    assert len(final_state["step_results"]) == 2
    assert final_state["step_results"][0].action_taken.startswith("fill ->")
    assert final_state["step_results"][1].success is True
    assert fake_browser.started_with == "https://example.com/login"
    assert fake_browser.stopped is True


@pytest.mark.asyncio
async def test_agent_fails_when_verifier_says_expected_result_not_met(
    sample_test_case, fake_browser
):
    llm = FakeLLM(
        responses=[
            '{"action": "fill", "selector": "input[name=\'username\']", "text": "demo_user"}',
            '{"action": "click", "selector": "button[type=\'submit\']"}',
            '{"passed": false, "reason": "Page still shows the login form, not the dashboard"}',
        ]
    )
    agent = QAAgent(browser=fake_browser, llm=llm)

    final_state = await agent.run(sample_test_case)

    assert final_state["passed"] is False
    assert "login form" in final_state["verdict_reason"]


@pytest.mark.asyncio
async def test_agent_short_circuits_to_verify_on_browser_action_failure(
    sample_test_case, fake_browser
):
    async def failing_click(selector: str):
        from app.browser_executor import ActionResult

        return ActionResult(False, f"click failed on {selector}: element not found")

    fake_browser.click = failing_click  # type: ignore[method-assign]

    llm = FakeLLM(
        responses=[
            '{"action": "fill", "selector": "input[name=\'username\']", "text": "demo_user"}',
            '{"action": "click", "selector": "button[type=\'submit\']"}',
        ]
    )
    agent = QAAgent(browser=fake_browser, llm=llm)

    final_state = await agent.run(sample_test_case)

    # Should skip the verifier LLM call entirely and fail on the error.
    assert final_state["passed"] is False
    assert "element not found" in final_state["verdict_reason"]
    assert llm.calls == 2  # only the two planner calls, no verifier call


@pytest.mark.asyncio
async def test_agent_handles_malformed_planner_json_gracefully(sample_test_case, fake_browser):
    llm = FakeLLM(
        responses=[
            "not valid json at all",
            '{"passed": false, "reason": "fallback action did nothing useful"}',
        ]
    )
    agent = QAAgent(browser=fake_browser, llm=llm)

    final_state = await agent.run(sample_test_case)

    # Rejects malformed planner output rather than executing an arbitrary action.
    assert final_state["step_results"][0].action_taken == "invalid -> [rejected]"


@pytest.mark.asyncio
async def test_agent_rejects_unknown_planner_action(sample_test_case, fake_browser):
    llm = FakeLLM(responses=['{"action": "evaluate", "selector": "body"}'])
    agent = QAAgent(browser=fake_browser, llm=llm)

    final_state = await agent.run(sample_test_case)

    assert final_state["passed"] is False
    assert final_state["step_results"][0].success is False
    assert fake_browser.calls == []
