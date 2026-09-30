"""Run the app with `python run.py` instead of the `uvicorn` CLI directly.

Why this file exists: on Windows, Playwright launches Chromium as a
subprocess, which asyncio can only do under the ProactorEventLoop. Calling
`uvicorn app.main:app --reload` from the CLI sets up uvicorn's event loop
before your app module (main.py) ever finishes importing — so a policy
change placed inside main.py is too late to take effect. Setting it here,
before uvicorn is even imported, guarantees it's in place from the start.
"""
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    # reload=False for now, deliberately: uvicorn's --reload on Windows
    # restarts the server in a way that can reintroduce exactly this
    # event-loop-policy problem. Once you've confirmed a real test run
    # passes, we can revisit turning reload back on. Until then, just
    # Ctrl+C and rerun `python run.py` after code changes.
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)