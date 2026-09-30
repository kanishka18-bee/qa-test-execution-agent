from __future__ import annotations

import asyncio
import ipaddress
import time
import uuid

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles

from app.agent import QAAgent
from app.browser_executor import BrowserExecutor
from app.config import get_settings
from app.schemas import RunStatus, StepResult, TestRunRequest, TestRunResult

app = FastAPI(title="QA Test-Execution Agent")
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=get_settings().allowed_hosts)

# In-memory store for the take-home scope. Swap for Redis/Postgres to persist
# across restarts or scale beyond one worker.
_RUNS: dict[str, TestRunResult] = {}
_RUNS_LOCK = asyncio.Lock()
_RUN_SEMAPHORE = asyncio.Semaphore(get_settings().max_concurrent_runs)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


async def _execute_run(run_id: str) -> None:
    async with _RUN_SEMAPHORE:
        result = _RUNS[run_id]
        result.status = RunStatus.RUNNING
        started = time.perf_counter()

        try:
            print("STARTING RUN:", run_id)

            settings = get_settings()
            browser = BrowserExecutor(headless=settings.headless_browser)
            # Constructing QAAgent builds the LLM client, which validates the
            # API key immediately — this must be inside the try block, or a
            # bad/missing key throws here and the run silently dies while
            # stuck at status=RUNNING forever (nothing ever catches it).
            agent = QAAgent(browser=browser)

            # QAAgent.run() owns the full browser lifecycle (start + stop
            # in its own try/finally) — do NOT call browser.start()/stop()
            # here too. Doing both launches two Chromium instances per run
            # and leaks the first one, since only the second reference
            # survives to be closed.
            final_state = await agent.run(result.test_case)

            result.step_results = [
                StepResult(**r.model_dump()) for r in final_state["step_results"]
            ]
            result.verdict_reason = final_state.get("verdict_reason", "")
            result.llm_calls = final_state.get("llm_calls", 0)
            result.input_tokens = final_state.get("input_tokens", 0)
            result.output_tokens = final_state.get("output_tokens", 0)
            result.used_fallback = final_state.get("used_fallback", False)
            result.status = RunStatus.PASSED if final_state.get("passed") else RunStatus.FAILED

            print("FINISHED RUN:", run_id, "->", result.status)

        except Exception as e:  # noqa: BLE001 - surface any failure as a run error
            print("ERROR in run", run_id, ":", str(e))
            result.status = RunStatus.ERROR
            result.error = str(e)

        finally:
            result.duration_seconds = round(time.perf_counter() - started, 2)


@app.post("/test-runs", response_model=TestRunResult, status_code=202)
async def create_test_run(
    request: TestRunRequest,
    background_tasks: BackgroundTasks,
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> TestRunResult:
    settings = get_settings()
    if settings.api_key and api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="authentication required")

    # Fast, shallow pre-check on literal IP/localhost hosts. The thorough
    # check (DNS resolution + public-IP check) happens in
    # BrowserExecutor._validate_url right before the browser actually
    # navigates there — this one just fails obviously-bad requests early.
    hostname = request.test_case.start_url.host.lower().rstrip(".")
    try:
        host_is_private = not ipaddress.ip_address(hostname).is_global
    except ValueError:
        host_is_private = hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local")
    if host_is_private:
        raise HTTPException(status_code=400, detail="start_url host is not allowed")

    run_id = str(uuid.uuid4())
    result = TestRunResult(run_id=run_id, status=RunStatus.QUEUED, test_case=request.test_case)
    async with _RUNS_LOCK:
        if len(_RUNS) >= settings.max_stored_runs:
            del _RUNS[next(iter(_RUNS))]
        _RUNS[run_id] = result
    background_tasks.add_task(_execute_run, run_id)
    return result


@app.get("/test-runs/{run_id}", response_model=TestRunResult)
async def get_test_run(
    run_id: str,
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> TestRunResult:
    settings = get_settings()
    if settings.api_key and api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="authentication required")
    result = _RUNS.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="run not found")
    return result


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
