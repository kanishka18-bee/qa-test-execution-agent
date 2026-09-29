"""Run this directly: python diagnose_gemini.py

Bypasses the agent, browser, and FastAPI entirely. Fires a few tiny, cheap
requests at each candidate model back to back and reports pass rate and
timing per model, so model choice is based on real numbers instead of a
guess. Useful on its own, and also feeds directly into the "model
selection" evidence the JD asks for.
"""
import os
import time

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

API_KEY = os.getenv("GEMINI_API_KEY", "")
N_CALLS_PER_MODEL = 5

# Candidates worth comparing: your current model, plus its close siblings
# with generous free-tier daily quota per the AI Studio rate-limit page.
MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]


def test_model(model: str) -> dict:
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=API_KEY, max_retries=0, timeout=15)
    successes = 0
    latencies = []
    print(f"\n{model}")
    for i in range(1, N_CALLS_PER_MODEL + 1):
        start = time.perf_counter()
        try:
            llm.invoke([HumanMessage(content="Reply with exactly one word: ok")])
            elapsed = time.perf_counter() - start
            print(f"  call {i}: OK in {elapsed:.2f}s")
            successes += 1
            latencies.append(elapsed)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            reason = str(exc).split(".", 1)[0]  # just the "503 UNAVAILABLE" / "504 ..." part
            print(f"  call {i}: FAILED after {elapsed:.2f}s -> {reason}")
        time.sleep(1)
    avg = sum(latencies) / len(latencies) if latencies else 0.0
    return {"model": model, "successes": successes, "avg_latency": avg}


results = [test_model(m) for m in MODELS]

print("\n" + "=" * 50)
print(f"{'Model':<26}{'Pass rate':<12}{'Avg latency (ok calls)'}")
for r in results:
    rate = f"{r['successes']}/{N_CALLS_PER_MODEL}"
    lat = f"{r['avg_latency']:.2f}s" if r["avg_latency"] else "n/a"
    print(f"{r['model']:<26}{rate:<12}{lat}")