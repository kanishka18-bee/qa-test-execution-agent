"""Pydantic models shared across the API and the agent."""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, Field, HttpUrl


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class TestCase(BaseModel):
    """Mirrors the shape of a Zephyr/Jira test case closely enough to swap
    in a real integration later without changing the agent."""

    title: str = Field(..., min_length=1, max_length=200, examples=["User can log in with valid credentials"])
    start_url: HttpUrl = Field(..., examples=["https://example.com/login"])
    steps: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(
        ...,
        min_length=1,
        max_length=25,
        examples=[[
            "Enter 'demo_user' into the username field",
            "Enter 'correct_password' into the password field",
            "Click the login button",
        ]],
    )
    expected_result: str = Field(..., min_length=1, max_length=2000)
    jira_issue_key: Optional[str] = Field(
        default=None, max_length=32, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$",
        description="e.g. QA-123, for correlating back to Jira/Zephyr"
    )

    model_config = {"str_strip_whitespace": True}


class Plan(BaseModel):
    action: str = Field(..., pattern=r"^(click|fill)$")
    selector: str = Field(..., min_length=1, max_length=500)
    text: str = Field(default="", max_length=2000)

    model_config = {"extra": "forbid", "str_strip_whitespace": True}


class TestRunRequest(BaseModel):
    test_case: TestCase


class StepResult(BaseModel):
    step: str
    action_taken: str
    success: bool
    detail: str = ""


class TestRunResult(BaseModel):
    run_id: str
    status: RunStatus
    test_case: TestCase
    step_results: list[StepResult] = Field(default_factory=list)
    verdict_reason: str = ""
    error: Optional[str] = None
