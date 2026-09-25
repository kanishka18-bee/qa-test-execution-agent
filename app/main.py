from __future__ import annotations

import asyncio
import ipaddress
import uuid

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.agent import QAAgent
from app.browser_executor import BrowserExecutor
from app.config import get_settings
from app.schemas import RunStatus, StepResult, TestRunRequest, TestRunResult

app = FastAPI(title="QA Test-Execution Agent")
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


# async def _execute_run(run_id: str) -> None:
#     async with _RUN_SEMAPHORE:
#         result = _RUNS[run_id]
#         result.status = RunStatus.RUNNING
#         settings = get_settings()
#         browser = BrowserExecutor(headless=settings.headless_browser)
#         agent = QAAgent(browser=browser)
#         try:
#             final_state = await agent.run(result.test_case)
#             result.step_results = [StepResult(**r.model_dump()) for r in final_state["step_results"]]
#             result.verdict_reason = final_state.get("verdict_reason", "")
#             result.status = RunStatus.PASSED if final_state.get("passed") else RunStatus.FAILED
#         except Exception:
#             result.status = RunStatus.ERROR
#             result.error = "test run failed; inspect server logs for details"

async def _execute_run(run_id: str) -> None:
    async with _RUN_SEMAPHORE:
        result = _RUNS[run_id]
        result.status = RunStatus.RUNNING

        settings = get_settings()
        browser = BrowserExecutor(headless=settings.headless_browser)
        agent = QAAgent(browser=browser)

        try:
            print("🚀 STARTING RUN:", run_id)

            # ✅ START browser
            await browser.start(result.test_case.start_url)

            final_state = await agent.run(result.test_case)

            result.step_results = [
                StepResult(**r.model_dump()) for r in final_state["step_results"]
            ]
            result.verdict_reason = final_state.get("verdict_reason", "")
            result.status = RunStatus.PASSED if final_state.get("passed") else RunStatus.FAILED

            print("✅ FINISHED RUN:", run_id)

        except Exception as e:
            print("❌ ERROR:", str(e))
            result.status = RunStatus.ERROR
            result.error = str(e)

        finally:
            # ✅ ALWAYS stop browser
            await browser.stop()

@app.post("/test-runs", response_model=TestRunResult, status_code=202)
async def create_test_run(
    request: TestRunRequest,
    background_tasks: BackgroundTasks,
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> TestRunResult:
    settings = get_settings()
    if settings.api_key and api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="authentication required")
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
