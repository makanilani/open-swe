from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from agent.utils.sandbox import (
    DEFAULT_MAX_CONCURRENT_CREATE,
    SANDBOX_MAX_CONCURRENT_CREATE_ENV_VAR,
    _get_create_semaphore,
    _validate_sandbox_max_concurrent,
    create_sandbox,
)


@pytest.fixture(autouse=True)
def _reset_global_semaphore() -> None:
    import agent.utils.sandbox as sandbox_mod

    sandbox_mod._create_semaphore = None


class TestGetCreateSemaphore:
    @pytest.mark.asyncio
    async def test_returns_same_instance_on_repeated_calls(self) -> None:
        sem_a = await _get_create_semaphore()
        sem_b = await _get_create_semaphore()
        assert sem_a is sem_b

    @pytest.mark.asyncio
    async def test_default_limit(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            sem = await _get_create_semaphore()
            assert sem._value == DEFAULT_MAX_CONCURRENT_CREATE

    @pytest.mark.asyncio
    async def test_custom_limit_from_env(self) -> None:
        with patch.dict(os.environ, {SANDBOX_MAX_CONCURRENT_CREATE_ENV_VAR: "8"}):
            sem = await _get_create_semaphore()
            assert sem._value == 8


class TestValidateMaxConcurrent:
    def test_accepts_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            _validate_sandbox_max_concurrent()

    def test_accepts_valid_int(self) -> None:
        with patch.dict(os.environ, {SANDBOX_MAX_CONCURRENT_CREATE_ENV_VAR: "10"}):
            _validate_sandbox_max_concurrent()

    def test_rejects_non_int(self) -> None:
        with patch.dict(os.environ, {SANDBOX_MAX_CONCURRENT_CREATE_ENV_VAR: "not-a-number"}):
            with pytest.raises(ValueError, match="must be an integer"):
                _validate_sandbox_max_concurrent()

    def test_rejects_zero(self) -> None:
        with patch.dict(os.environ, {SANDBOX_MAX_CONCURRENT_CREATE_ENV_VAR: "0"}):
            with pytest.raises(ValueError, match=">= 1"):
                _validate_sandbox_max_concurrent()

    def test_rejects_negative(self) -> None:
        with patch.dict(os.environ, {SANDBOX_MAX_CONCURRENT_CREATE_ENV_VAR: "-1"}):
            with pytest.raises(ValueError, match=">= 1"):
                _validate_sandbox_max_concurrent()


class TestCreateSandboxSemaphore:
    @pytest.mark.asyncio
    async def test_acquires_semaphore_before_factory(self) -> None:
        mock_factory = MagicMock(return_value=MagicMock(id="sandbox-1"))
        with (
            patch.dict("agent.utils.sandbox.SANDBOX_FACTORIES", {"test": mock_factory}),
            patch.dict(
                os.environ,
                {"SANDBOX_TYPE": "test", SANDBOX_MAX_CONCURRENT_CREATE_ENV_VAR: "1"},
            ),
        ):
            result = await create_sandbox(sandbox_id="test-sandbox")
            assert result.id == "sandbox-1"
            mock_factory.assert_called_once_with("test-sandbox")

    @pytest.mark.asyncio
    async def test_limits_concurrent_creations(self) -> None:
        started = 0
        max_concurrent = 0
        start_lock = asyncio.Lock()
        proceed = asyncio.Event()

        async def slow_factory(
            sandbox_id: str | None = None,
            github_token: str | None = None,
        ) -> MagicMock:
            nonlocal started, max_concurrent
            async with start_lock:
                started += 1
                max_concurrent = max(max_concurrent, started)
            await proceed.wait()
            async with start_lock:
                started -= 1
            return MagicMock(id=f"sandbox-{sandbox_id}")

        with (
            patch.dict("agent.utils.sandbox.SANDBOX_FACTORIES", {"test": slow_factory}),
            patch.dict(
                os.environ,
                {"SANDBOX_TYPE": "test", SANDBOX_MAX_CONCURRENT_CREATE_ENV_VAR: "2"},
            ),
        ):
            tasks = [asyncio.create_task(create_sandbox(sandbox_id=str(i))) for i in range(4)]
            await asyncio.sleep(0.1)
            assert max_concurrent <= 2, f"Expected <=2 concurrent, got {max_concurrent}"
            proceed.set()
            results = await asyncio.gather(*tasks)
            assert len(results) == 4
