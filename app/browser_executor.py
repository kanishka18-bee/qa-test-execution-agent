"""Thin async wrapper around Playwright (which drives Chrome over CDP).

Kept deliberately small and dependency-injectable so it can be mocked in
tests without spinning up a real browser.
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from playwright.async_api import Browser, Page, async_playwright

from app.config import get_settings


@dataclass
class ActionResult:
    success: bool
    detail: str


class BrowserExecutor:
    """One instance = one test run's browser session.

    Owns the full Playwright lifecycle (start -> actions -> stop). Callers
    should call start() exactly once and stop() exactly once — QAAgent.run()
    already does this; don't also call start()/stop() around it, or you'll
    launch two browsers per run and leak the first one.
    """

    def __init__(self, headless: bool = True) -> None:
        self._headless = headless
        self._settings = get_settings()
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None

    async def start(self, start_url: str) -> None:
        start_url = str(start_url)
        self._validate_url(start_url)
        self._playwright = await async_playwright().start()
        try:
            self._browser = await self._playwright.chromium.launch(headless=self._headless)
            context = await self._browser.new_context(service_workers="block")
            self._page = await context.new_page()
            self._page.set_default_timeout(self._settings.browser_timeout_ms)
            self._page.set_default_navigation_timeout(self._settings.browser_timeout_ms)
            await self._page.goto(start_url, wait_until="domcontentloaded")
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        try:
            if self._browser:
                await self._browser.close()
        finally:
            self._browser = None
            self._page = None
            if self._playwright:
                await self._playwright.stop()
            self._playwright = None

    def _ensure_page(self) -> Page:
        if not self._page:
            raise RuntimeError("Browser page not initialized. Call start() first.")
        return self._page

    async def click(self, selector: str) -> ActionResult:
        page = self._ensure_page()
        try:
            await page.wait_for_selector(selector, timeout=self._settings.browser_timeout_ms)
            await page.click(selector, timeout=self._settings.browser_timeout_ms)
            # Give a possible navigation a moment to settle, but don't let it
            # eat the whole budget if nothing was actually navigating.
            await page.wait_for_load_state("domcontentloaded", timeout=min(5000, self._settings.browser_timeout_ms))
            await self._screenshot("click.png")
            return ActionResult(True, f"clicked {selector}")
        except Exception as exc:  # noqa: BLE001 - surface any Playwright error uniformly
            return ActionResult(False, f"click failed on {selector}: {exc}")

    async def fill(self, selector: str, text: str) -> ActionResult:
        page = self._ensure_page()
        try:
            await page.wait_for_selector(selector, timeout=self._settings.browser_timeout_ms)
            await page.fill(selector, text, timeout=self._settings.browser_timeout_ms)
            await self._screenshot("fill.png")
            return ActionResult(True, f"filled {selector} with '{text}'")
        except Exception as exc:  # noqa: BLE001
            return ActionResult(False, f"fill failed on {selector}: {exc}")

    async def get_visible_text(self) -> str:
        """Cheap page snapshot handed to the LLM for planning/verification."""
        page = self._ensure_page()
        try:
            await page.wait_for_load_state("domcontentloaded")
            body = await page.inner_text("body")
            return body[:3000]
        except Exception as exc:  # noqa: BLE001
            return f"[could not read page text: {exc}]"

    async def current_url(self) -> str:
        return self._page.url if self._page else ""

    async def _screenshot(self, name: str) -> None:
        if not self._settings.screenshots_enabled:
            return
        try:
            page = self._ensure_page()
            await page.screenshot(path=f"screenshots/{name}", animations="disabled")
        except Exception:  # noqa: BLE001 - screenshots are a nice-to-have, never fail the run over one
            pass

    def _validate_url(self, url: str) -> None:
        """Basic SSRF guard: only http(s), no embedded creds, must resolve to
        a public IP (or be explicitly allowlisted via ALLOWED_URL_HOSTS)."""
        url = str(url)
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            raise ValueError("start_url must be an HTTP(S) URL without embedded credentials")
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if not hostname:
            raise ValueError("start_url must include a hostname")
        if self._settings.allowed_url_hosts and not any(
            hostname == allowed or hostname.endswith(f".{allowed}")
            for allowed in self._settings.allowed_url_hosts
        ):
            raise ValueError("start_url host is not allowlisted")
        try:
            addresses = {info[4][0] for info in socket.getaddrinfo(hostname, None)}
        except socket.gaierror as exc:
            raise ValueError("start_url host could not be resolved") from exc
        for address in addresses:
            if not ipaddress.ip_address(address).is_global:
                raise ValueError("start_url must resolve to a public address")
