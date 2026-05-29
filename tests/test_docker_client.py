"""Unit tests for the shared async Docker client pool (docker_client.py)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_globals() -> None:
    """Reset module-level globals before each test to avoid cross-test leakage."""
    import agent.integrations.docker_client as dc

    dc._client = None
    dc._client_ref_count = 0
    dc._image_pull_locks = {}


@pytest.fixture
def mock_docker() -> MagicMock:
    """Mock ``aiodocker.Docker`` so no real daemon connection is made."""
    with patch("agent.integrations.docker_client.aiodocker.Docker") as m:
        m.return_value = AsyncMock()
        yield m


# ===========================================================================
# _get_docker_client
# ===========================================================================


class TestGetDockerClient:
    async def test_creates_client_on_first_call(self, mock_docker) -> None:
        import agent.integrations.docker_client as dc

        client = await dc._get_docker_client()

        mock_docker.assert_called_once()
        assert client is mock_docker.return_value

    async def test_returns_singleton(self, mock_docker) -> None:
        import agent.integrations.docker_client as dc

        c1 = await dc._get_docker_client()
        c2 = await dc._get_docker_client()

        assert c1 is c2
        mock_docker.assert_called_once()

    async def test_increments_ref_count(self, mock_docker) -> None:
        import agent.integrations.docker_client as dc

        assert dc._client_ref_count == 0

        await dc._get_docker_client()
        assert dc._client_ref_count == 1

        await dc._get_docker_client()
        assert dc._client_ref_count == 2

    async def test_passes_docker_host_from_env(self, mock_docker) -> None:
        import agent.integrations.docker_client as dc

        with patch.dict(os.environ, {"DOCKER_HOST": "tcp://localhost:2376"}, clear=True):
            await dc._get_docker_client()

        mock_docker.assert_called_once_with(url="tcp://localhost:2376")

    async def test_no_arg_when_docker_host_empty(self, mock_docker) -> None:
        import agent.integrations.docker_client as dc

        with patch.dict(os.environ, {}, clear=True):
            await dc._get_docker_client()

        mock_docker.assert_called_once_with()


# ===========================================================================
# _release_docker_client
# ===========================================================================


class TestReleaseDockerClient:
    async def test_decrements_ref_count(self, mock_docker) -> None:
        import agent.integrations.docker_client as dc

        await dc._get_docker_client()
        await dc._get_docker_client()
        assert dc._client_ref_count == 2

        await dc._release_docker_client()
        assert dc._client_ref_count == 1

    async def test_closes_client_when_count_reaches_zero(self, mock_docker) -> None:
        import agent.integrations.docker_client as dc

        client = await dc._get_docker_client()

        await dc._release_docker_client()

        assert dc._client is None
        assert dc._client_ref_count == 0
        client.close.assert_awaited_once()

    async def test_close_on_zero_handles_exception(self, mock_docker) -> None:
        import agent.integrations.docker_client as dc

        client = await dc._get_docker_client()
        client.close.side_effect = RuntimeError("close failed")

        await dc._release_docker_client()

        assert dc._client is None
        assert dc._client_ref_count == 0

    async def test_no_guard_below_zero(self, mock_docker) -> None:
        import agent.integrations.docker_client as dc

        await dc._release_docker_client()
        assert dc._client_ref_count == -1
        assert dc._client is None


# ===========================================================================
# _get_image_pull_lock
# ===========================================================================


class TestGetImagePullLock:
    def test_returns_lock_for_image(self) -> None:
        import agent.integrations.docker_client as dc

        lock = dc._get_image_pull_lock("open-swe-sandbox:latest")

        assert isinstance(lock, dc.asyncio.Lock)

    def test_same_lock_for_same_image(self) -> None:
        import agent.integrations.docker_client as dc

        lock1 = dc._get_image_pull_lock("img:1")
        lock2 = dc._get_image_pull_lock("img:1")

        assert lock1 is lock2

    def test_different_locks_for_different_images(self) -> None:
        import agent.integrations.docker_client as dc

        lock1 = dc._get_image_pull_lock("img:1")
        lock2 = dc._get_image_pull_lock("img:2")

        assert lock1 is not lock2

    def test_thread_safe_storage(self) -> None:
        import agent.integrations.docker_client as dc

        lock1 = dc._get_image_pull_lock("img")
        assert dc._image_pull_locks["img"] is lock1


# ===========================================================================
# _ensure_image
# ===========================================================================


class TestEnsureImage:
    async def test_inspects_image_locally_first(self, mock_docker) -> None:
        import agent.integrations.docker_client as dc

        client = await dc._get_docker_client()

        await dc._ensure_image(client, "open-swe-sandbox:latest")

        client.images.inspect.assert_awaited_once_with("open-swe-sandbox:latest")
        client.images.pull.assert_not_awaited()

    async def test_pulls_when_image_not_found(self, mock_docker) -> None:
        import agent.integrations.docker_client as dc

        client = await dc._get_docker_client()
        client.images.inspect.side_effect = [
            dc.aiodocker.exceptions.DockerError(404, {"message": "not found"}),
            dc.aiodocker.exceptions.DockerError(404, {"message": "not found"}),
        ]

        await dc._ensure_image(client, "open-swe-sandbox:latest")

        assert client.images.inspect.await_count == 2
        client.images.pull.assert_awaited_once_with("open-swe-sandbox:latest")

    async def test_double_check_under_lock_avoids_duplicate_pull(self, mock_docker) -> None:
        import agent.integrations.docker_client as dc

        client = await dc._get_docker_client()

        async def _inspect_side_effect(_image: str) -> None:
            raise dc.aiodocker.exceptions.DockerError(404, {"message": "not found"})

        client.images.inspect.side_effect = _inspect_side_effect

        await dc._ensure_image(client, "img:1")

        assert client.images.inspect.await_count == 2
        client.images.pull.assert_awaited_once_with("img:1")

    async def test_docker_error_during_pull_propagates(self, mock_docker) -> None:
        import agent.integrations.docker_client as dc

        client = await dc._get_docker_client()
        client.images.inspect.side_effect = [
            dc.aiodocker.exceptions.DockerError(404, {"message": "not found"}),
            dc.aiodocker.exceptions.DockerError(404, {"message": "not found"}),
        ]
        client.images.pull.side_effect = dc.aiodocker.exceptions.DockerError(
            500, {"message": "pull failed"}
        )

        with pytest.raises(dc.aiodocker.exceptions.DockerError):
            await dc._ensure_image(client, "img:1")


# ===========================================================================
# _force_close_client / _atexit_close
# ===========================================================================


class TestForceCloseClient:
    async def test_closes_existing_client(self, mock_docker) -> None:
        import agent.integrations.docker_client as dc

        client = await dc._get_docker_client()

        await dc._force_close_client()

        assert dc._client is None
        assert dc._client_ref_count == 0
        client.close.assert_awaited_once()

    async def test_noop_when_no_client(self) -> None:
        import agent.integrations.docker_client as dc

        await dc._force_close_client()

        assert dc._client is None

    async def test_handles_close_exception(self, mock_docker) -> None:
        import agent.integrations.docker_client as dc

        client = await dc._get_docker_client()
        client.close.side_effect = RuntimeError("close error")

        await dc._force_close_client()

        assert dc._client is None
        assert dc._client_ref_count == 0
