import asyncio

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app
from app.schemas import RunStatus, StepResult

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_runs():
    main_module._RUNS.clear()
    yield
    main_module._RUNS.clear()


def test_create_test_run_returns_queued_immediately(sample_test_case, monkeypatch):
    async def fake_execute_run(run_id: str) -> None:
        # Simulate the background task doing nothing (yet) so we can assert
        # the immediate response without racing a real agent run.
        return None

    monkeypatch.setattr(main_module, "_execute_run", fake_execute_run)

    response = client.post(
        "/test-runs",
        json={"test_case": sample_test_case.model_dump(mode="json")},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == RunStatus.QUEUED.value
    assert body["test_case"]["title"] == sample_test_case.title
    assert "run_id" in body


def test_get_test_run_returns_404_for_unknown_id():
    response = client.get("/test-runs/does-not-exist")
    assert response.status_code == 404


def test_full_flow_reaches_passed_status(sample_test_case, monkeypatch):
    async def fake_execute_run(run_id: str) -> None:
        result = main_module._RUNS[run_id]
        result.status = RunStatus.PASSED
        result.step_results = [
            StepResult(step=sample_test_case.steps[0], action_taken="fill -> input", success=True)
        ]
        result.verdict_reason = "dashboard shown as expected"

    monkeypatch.setattr(main_module, "_execute_run", fake_execute_run)

    create_response = client.post(
        "/test-runs",
        json={"test_case": sample_test_case.model_dump(mode="json")},
    )
    run_id = create_response.json()["run_id"]

    get_response = client.get(f"/test-runs/{run_id}")
    body = get_response.json()

    assert body["status"] == RunStatus.PASSED.value
    assert body["verdict_reason"] == "dashboard shown as expected"
    assert len(body["step_results"]) == 1


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_api_key_is_required_when_configured(sample_test_case, monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret")
    main_module.get_settings.cache_clear()

    response = client.post(
        "/test-runs",
        json={"test_case": sample_test_case.model_dump(mode="json")},
    )

    assert response.status_code == 401
    main_module.get_settings.cache_clear()


def test_local_start_url_is_rejected(sample_test_case):
    payload = {"test_case": sample_test_case.model_dump(mode="json")}
    payload["test_case"]["start_url"] = "http://127.0.0.1:8000/admin"

    response = client.post("/test-runs", json=payload)

    assert response.status_code == 400
