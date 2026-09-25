# QA Test-Execution Agent

A LangGraph agent that takes a natural-language test case (the shape a
Jira/Zephyr ticket would have), drives a real browser via Playwright/CDP to
execute the steps, and verifies the outcome against the expected result —
exposed over an async FastAPI service.

## Why this exists

Built as a portfolio project targeting roles that combine agent
orchestration (LangGraph/LangChain) with browser automation and QA tooling
(CDP, Jira/Zephyr, Jenkins). It's scoped to be runnable and testable in a
day, not a production QA platform — see "Not included" below for what a
real version would add.

## Architecture

```
POST /test-runs  ──►  in-memory run store  ──►  background task
                                                       │
                                                       ▼
                                              QAAgent.run(test_case)
                                                       │
                              ┌────────────────────────┴───────────────────┐
                              │              LangGraph graph                │
                              │                                             │
                              │   plan_step ──► execute_step ──► (loop)     │
                              │       ▲               │                    │
                              │       └───────────────┘                    │
                              │                        │                   │
                              │                   (steps done/error)       │
                              │                        ▼                   │
                              │                     verify ──► END         │
                              └─────────────────────────────────────────────┘
                                       │                    │
                                 Gemini (plan/verify)   Playwright (act)
```

- **`plan_step`** — sends the current test step + the page's visible text to
  Gemini, gets back a single `{action, selector, text}` JSON instruction.
- **`execute_step`** — runs that action through `BrowserExecutor`
  (Playwright, which talks to Chromium over CDP), appends a `StepResult`,
  and loops back to `plan_step` for the next test step.
- **`verify`** — once all steps are done (or a step failed), asks Gemini to
  compare the final page text against `expected_result` and returns a
  pass/fail verdict with a reason.
- **`GET /test-runs/{id}`** — poll for status, per-step results, and the
  verdict.

## Running locally

```bash
cp .env.example .env   # add your GEMINI_API_KEY
docker compose up --build
```

Then:

```bash
curl -X POST http://localhost:8000/test-runs \
  -H "Content-Type: application/json" \
  -d '{
    "test_case": {
      "title": "User can log in with valid credentials",
      "start_url": "https://example.com/login",
      "steps": [
        "Enter '\''demo_user'\'' into the username field",
        "Click the login button"
      ],
      "expected_result": "The dashboard page is shown with a welcome message",
      "jira_issue_key": "QA-123"
    }
  }'

curl http://localhost:8000/test-runs/<run_id>
```

For a shared deployment, set `API_KEY` and send it as `X-API-Key`. Set
`ALLOWED_URL_HOSTS` to the exact target domains the browser may visit. The
service rejects credentials in URLs and non-public targets to reduce SSRF
risk, and runs as a non-root user in the container.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

The test suite mocks both the LLM and the browser (see `tests/conftest.py`)
so it runs in under a second with no API key, no network access, and no
real Chromium instance — the graph's control flow (looping over steps,
short-circuiting to `verify` on a browser error, falling back gracefully on
malformed LLM output) is what's under test.

## Not included (by design, given the timebox)

- Persistent run storage (currently an in-memory dict — swap for
  Redis/Postgres to survive restarts or scale past one worker)
- Real Jira/Zephyr and Jenkins integration (the `TestCase` schema is shaped
  to match one, but pulling/pushing is stubbed out)
- Retry/backoff on flaky selectors, richer DOM-aware selector inference,
  screenshots-on-failure
- Persistent storage, distributed queueing, and rate limiting (put these
  behind Redis/Postgres and an API gateway for a multi-worker deployment)
- Structured logging/observability and real Jira/Zephyr integration
