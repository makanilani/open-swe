"""Unit tests for the Docker sandbox backend.

The ``run_async`` bridge is mocked so no real Docker daemon is needed.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)

from agent.integrations.docker_sandbox import (
    DEFAULT_CONTAINER_PREFIX,
    DockerSandbox,
    _build_container_name,
    _build_image_name,
    _parse_mem,
    create_docker_sandbox,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def fake_run_async():
    """Mock ``run_async`` in both modules so no Docker daemon is contacted."""
    with patch("agent.integrations.docker_sandbox.run_async") as m:
        import agent.integrations.docker as _d

        _d.run_async = m
        yield m


# ===========================================================================
# Helpers
# ===========================================================================


class TestBuildImageName:
    def test_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _build_image_name() == "open-swe-sandbox:latest"

    def test_from_env(self) -> None:
        with patch.dict(os.environ, {"DOCKER_SANDBOX_IMAGE": "my-registry/swe:1.0"}, clear=True):
            assert _build_image_name() == "my-registry/swe:1.0"


class TestBuildContainerName:
    def test_default_prefix(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            name = _build_container_name()
            assert name.startswith(DEFAULT_CONTAINER_PREFIX)
            suffix = name[len(DEFAULT_CONTAINER_PREFIX) :]
            assert len(suffix) == 8
            assert suffix.isalnum()

    def test_custom_prefix(self) -> None:
        with patch.dict(os.environ, {"DOCKER_CONTAINER_PREFIX": "my-swe-"}):
            name = _build_container_name()
            assert name.startswith("my-swe-")
            assert len(name) == len("my-swe-") + 8


class TestParseMem:
    def test_bytes(self) -> None:
        assert _parse_mem("536870912") == 536870912

    def test_gigabytes(self) -> None:
        assert _parse_mem("4g") == 4 * 10**9

    def test_megabytes(self) -> None:
        assert _parse_mem("512m") == 512 * 10**6

    def test_kilobytes(self) -> None:
        assert _parse_mem("128k") == 128 * 10**3

    def test_case_insensitive(self) -> None:
        assert _parse_mem("2G") == 2 * 10**9


# ===========================================================================
# DockerSandbox
# ===========================================================================


class TestDockerSandboxId:
    def test_returns_short_id(self) -> None:
        sb = DockerSandbox("abc123def456ghi789")
        assert sb.id == "abc123def456"

    def test_handles_short_ids(self) -> None:
        sb = DockerSandbox("abc")
        assert sb.id == "abc"


class TestDockerSandboxExecute:
    def test_returns_output_and_exit_code(self, fake_run_async) -> None:
        fake_run_async.return_value = ("hello world", 0)
        sb = DockerSandbox("c1")

        result = sb.execute("echo hello")

        assert isinstance(result, ExecuteResponse)
        assert result.output == "hello world"
        assert result.exit_code == 0
        assert result.truncated is False

    def test_propagates_timeout(self, fake_run_async) -> None:
        fake_run_async.return_value = ("", 0)
        sb = DockerSandbox("c1")
        result = sb.execute("sleep 10", timeout=5)
        assert result.exit_code == 0
        _call_args = fake_run_async.call_args
        assert _call_args is not None
        _kwargs = _call_args[1]
        assert _kwargs.get("timeout") == 5

    def test_returns_timeout_error_on_timeout(self, fake_run_async) -> None:
        fake_run_async.side_effect = TimeoutError("timed out")
        sb = DockerSandbox("c1")

        result = sb.execute("sleep 100")

        assert isinstance(result, ExecuteResponse)
        assert "timed out" in result.output
        assert result.exit_code == -1

    def test_closed_sandbox_returns_error(self, fake_run_async) -> None:
        sb = DockerSandbox("c1")
        sb._closed = True

        result = sb.execute("echo hello")

        assert result.exit_code == 1
        assert result.output == ""
        fake_run_async.assert_not_called()

    def test_truncated_flag(self, fake_run_async) -> None:
        fake_run_async.return_value = ("x" * 600_000, 0)
        sb = DockerSandbox("c1")

        result = sb.execute("cat large_file")

        assert result.truncated is True

    def test_default_timeout_from_env(self, fake_run_async) -> None:
        fake_run_async.return_value = ("", 0)
        with patch.dict(os.environ, {"DOCKER_SANDBOX_TIMEOUT": "120"}):
            sb = DockerSandbox("c1")
            sb.execute("echo hi")
            _kwargs = fake_run_async.call_args[1]
            assert _kwargs.get("timeout") == 120


class TestDockerSandboxUploadFiles:
    def test_returns_upload_responses(self, fake_run_async) -> None:
        fake_run_async.return_value = [
            FileUploadResponse(path="/workspace/foo.py"),
            FileUploadResponse(path="/workspace/bar.py"),
        ]
        sb = DockerSandbox("c1")

        files = [("/workspace/foo.py", b"content1"), ("/workspace/bar.py", b"content2")]
        result = sb.upload_files(files)

        assert len(result) == 2
        assert result[0].path == "/workspace/foo.py"
        assert result[0].error is None
        assert result[1].path == "/workspace/bar.py"
        assert result[1].error is None

    def test_reports_partial_failures(self, fake_run_async) -> None:
        fake_run_async.return_value = [
            FileUploadResponse(path="/workspace/ok.py"),
            FileUploadResponse(path="/workspace/fail.py", error="permission_denied"),
        ]
        sb = DockerSandbox("c1")

        files = [("/workspace/ok.py", b"ok"), ("/workspace/fail.py", b"fail")]
        result = sb.upload_files(files)

        assert result[0].error is None
        assert result[1].error == "permission_denied"


class TestDockerSandboxDownloadFiles:
    def test_returns_download_responses(self, fake_run_async) -> None:
        fake_run_async.return_value = [
            FileDownloadResponse(path="/workspace/foo.py", content=b"hello"),
            FileDownloadResponse(path="/workspace/missing.py", error="file_not_found"),
        ]
        sb = DockerSandbox("c1")

        result = sb.download_files(["/workspace/foo.py", "/workspace/missing.py"])

        assert result[0].path == "/workspace/foo.py"
        assert result[0].content == b"hello"
        assert result[1].path == "/workspace/missing.py"
        assert result[1].error == "file_not_found"


# ===========================================================================
# Lifecycle
# ===========================================================================


class TestDockerSandboxStart:
    def test_creates_container_and_waits_for_health(self, fake_run_async) -> None:
        """start() with no existing container should build image, create
        container, wait for health, and configure git credentials."""
        cid = "abcdef1234567890abcdef1234567890abcdef12"
        fake_run_async.side_effect = [
            None,  # _ensure_image_exists
            None,  # _find_existing_container → None (create new)
            cid,  # _create_and_start_container
            None,  # _wait_for_healthy
            ("", 0),  # _configure_git_credentials
        ]

        sb = DockerSandbox.start(gh_token="ghp_test")

        assert sb._container_id == cid
        assert sb._closed is False
        assert fake_run_async.call_count == 5

    def test_start_without_token_skips_git_config(self, fake_run_async) -> None:
        cid = "abc123"
        fake_run_async.side_effect = [
            None,  # _ensure_image_exists
            None,  # _find_existing_container → None (create new)
            cid,  # _create_and_start_container
            None,  # _wait_for_healthy
        ]

        sb = DockerSandbox.start(gh_token=None)

        assert sb._container_id == cid
        assert fake_run_async.call_count == 4

    def test_start_reconnects_existing_container(self, fake_run_async) -> None:
        """start() should reconnect to an existing container when found."""
        existing_id = "existing-abcdef123456"
        fake_run_async.side_effect = [
            None,  # _ensure_image_exists
            existing_id,  # _find_existing_container → found
            None,  # _ensure_container_running
        ]

        sb = DockerSandbox.start(gh_token=None)

        assert sb._container_id == existing_id
        assert sb._closed is False
        assert fake_run_async.call_count == 3

    def test_start_reconnects_with_git_config(self, fake_run_async) -> None:
        """start() should configure git credentials on reconnected container."""
        existing_id = "reused-container-id"
        fake_run_async.side_effect = [
            None,  # _ensure_image_exists
            existing_id,  # _find_existing_container → found
            None,  # _ensure_container_running
            ("", 0),  # _configure_git_credentials
        ]

        sb = DockerSandbox.start(gh_token="ghp_abc")

        assert sb._container_id == existing_id
        assert fake_run_async.call_count == 4


class TestDockerSandboxReconnect:
    def test_ensures_container_running(self, fake_run_async) -> None:
        fake_run_async.return_value = None  # _ensure_container_running
        sb = DockerSandbox.reconnect("existing-container-id")

        assert sb._container_id == "existing-container-id"
        assert sb._closed is False
        fake_run_async.assert_called_once()


class TestDockerSandboxClose:
    def test_stops_and_removes_container(self, fake_run_async) -> None:
        fake_run_async.return_value = None
        sb = DockerSandbox("c1")
        assert sb._closed is False

        sb.close()

        assert sb._closed is True
        fake_run_async.assert_called_once()

    def test_idempotent(self, fake_run_async) -> None:
        sb = DockerSandbox("c1")
        sb.close()
        fake_run_async.reset_mock()
        sb.close()
        fake_run_async.assert_not_called()


# ===========================================================================
# Factory
# ===========================================================================


class TestCreateDockerSandbox:
    async def test_new_sandbox(self, fake_run_async) -> None:
        """Should delegate to ``DockerSandboxBackend.start`` when no sandbox_id."""
        from agent.integrations.docker import DockerSandboxBackend

        with patch.object(DockerSandboxBackend, "start", return_value=None):
            sb = await create_docker_sandbox(sandbox_id=None, github_token="ghp_xxx")

            assert isinstance(sb, DockerSandboxBackend)
            assert sb._container_id is None

    async def test_reconnect(self, fake_run_async) -> None:
        """Should set container_id and ensure container running."""
        from agent.integrations.docker import DockerSandboxBackend

        fake_run_async.return_value = None  # _ensure_container_running
        sb = await create_docker_sandbox(sandbox_id="existing-id")

        assert isinstance(sb, DockerSandboxBackend)
        assert sb._container_id == "existing-id"
        fake_run_async.assert_called_once()


# ===========================================================================
# Sandbox error handling
# ===========================================================================


# ===========================================================================
# DockerSandboxBackend
# ===========================================================================


class TestDockerSandboxBackendId:
    def test_returns_empty_when_no_container(self) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        sb = DockerSandboxBackend(DockerSandboxConfig())
        assert sb.id == ""

    def test_returns_short_container_id(self) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        sb = DockerSandboxBackend(DockerSandboxConfig())
        sb._container_id = "abc123def456ghi789"
        assert sb.id == "abc123def456"

    def test_handles_short_ids(self) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        sb = DockerSandboxBackend(DockerSandboxConfig())
        sb._container_id = "abc"
        assert sb.id == "abc"


class TestDockerSandboxBackendExecute:
    def test_closed_returns_error(self, fake_run_async) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        sb = DockerSandboxBackend(DockerSandboxConfig())
        result = sb.execute("echo hello")
        assert result.exit_code == 1
        assert result.output == ""

    def test_no_container_id_returns_error(self, fake_run_async) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        sb = DockerSandboxBackend(DockerSandboxConfig())
        result = sb.execute("echo hello")
        assert result.exit_code == 1

    def test_returns_output_and_exit_code(self, fake_run_async) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        fake_run_async.return_value = ("hello world", 0)
        sb = DockerSandboxBackend(DockerSandboxConfig())
        sb._container_id = "c1"

        result = sb.execute("echo hello")
        assert result.output == "hello world"
        assert result.exit_code == 0
        assert result.truncated is False

    def test_returns_timeout_error(self, fake_run_async) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        fake_run_async.side_effect = TimeoutError("timed out")
        sb = DockerSandboxBackend(DockerSandboxConfig())
        sb._container_id = "c1"

        result = sb.execute("sleep 100")
        assert "timed out" in result.output
        assert result.exit_code == -1

    def test_truncated_flag(self, fake_run_async) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        fake_run_async.return_value = ("x" * 600_000, 0)
        sb = DockerSandboxBackend(DockerSandboxConfig())
        sb._container_id = "c1"

        result = sb.execute("cat large_file")
        assert result.truncated is True


class TestDockerSandboxBackendUploadDownload:
    def test_closed_upload_returns_error(self, fake_run_async) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        sb = DockerSandboxBackend(DockerSandboxConfig())
        result = sb.upload_files([("/f", b"data")])
        assert result[0].error == "sandbox closed"

    def test_closed_download_returns_error(self, fake_run_async) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        sb = DockerSandboxBackend(DockerSandboxConfig())
        result = sb.download_files(["/f"])
        assert result[0].error == "sandbox closed"


class TestDockerSandboxBackendClose:
    def test_closes_container(self, fake_run_async) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        fake_run_async.return_value = None
        sb = DockerSandboxBackend(DockerSandboxConfig())
        sb._container_id = "c1"
        assert sb._closed is False

        sb.close()
        assert sb._closed is True
        fake_run_async.assert_called_once()

    def test_idempotent(self, fake_run_async) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        sb = DockerSandboxBackend(DockerSandboxConfig())
        sb._container_id = "c1"
        sb.close()
        fake_run_async.reset_mock()
        sb.close()
        fake_run_async.assert_not_called()


class TestDockerSandboxBackendContainerName:
    def test_format(self) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        sb = DockerSandboxBackend(DockerSandboxConfig())
        name = sb._generate_container_name()
        assert name.startswith("open-swe-")
        parts = name.split("-")
        assert len(parts) == 4  # open, swe, {short_id}, {uuid}
        assert len(parts[2]) == 8
        assert len(parts[3]) == 32

    def test_unique_across_calls(self) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        sb = DockerSandboxBackend(DockerSandboxConfig())
        names = {sb._generate_container_name() for _ in range(100)}
        assert len(names) == 100


class TestDockerSandboxBackendConfig:
    def test_contains_required_fields(self) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        sb = DockerSandboxBackend(DockerSandboxConfig())
        config = sb._build_container_config()

        assert config["Image"] == "open-swe-sandbox:latest"
        assert config["Cmd"] == ["tail", "-f", "/dev/null"]
        assert config["WorkingDir"] == "/workspace"
        assert config["User"] == "swe-user"

    def test_sets_all_labels(self) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        sb = DockerSandboxBackend(DockerSandboxConfig())
        config = sb._build_container_config()
        labels = config["Labels"]

        assert labels["open-swe-sandbox"] == "true"
        assert labels["open-swe-managed"] == "true"
        assert "open-swe-created-at" in labels
        assert "open-swe-version" in labels

    def test_host_config_has_cap_drop_and_security(self) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        sb = DockerSandboxBackend(DockerSandboxConfig())
        config = sb._build_container_config()
        hc = config["HostConfig"]

        assert "NET_RAW" in hc["CapDrop"]
        assert "NET_ADMIN" in hc["CapDrop"]
        assert "SYS_ADMIN" in hc["CapDrop"]
        assert hc["ReadonlyRootfs"] is True
        assert "no-new-privileges:true" in hc["SecurityOpt"]

    def test_host_config_with_resources(self) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        cfg = DockerSandboxConfig(mem_limit="2g", cpu_limit=2.0, pids_limit=100)
        sb = DockerSandboxBackend(cfg)
        config = sb._build_container_config()
        hc = config["HostConfig"]

        assert hc["Memory"] == 2 * 10**9
        assert hc["NanoCpus"] == int(2.0 * 1e9)
        assert hc["PidsLimit"] == 100


class TestDockerSandboxBackendInjectSecrets:
    async def test_skips_when_no_secrets(self) -> None:
        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        sb = DockerSandboxBackend(DockerSandboxConfig())
        container = object()  # will not be accessed
        await sb._inject_secrets(container)  # should not raise

    async def test_injects_github_token(self) -> None:
        from unittest.mock import AsyncMock

        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        sb = DockerSandboxBackend(DockerSandboxConfig())
        container = AsyncMock()
        container._id = "c1"
        await sb._inject_secrets(container, github_token="ghp_secret")

        container.put_archive.assert_called_once()
        call_args = container.put_archive.call_args
        assert call_args[1]["path"] == "/"
        tar_data = call_args[1]["data"]
        assert b"GH_TOKEN" in tar_data
        assert b"ghp_secret" in tar_data

    async def test_injects_proxy_url(self) -> None:
        from unittest.mock import AsyncMock

        from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

        sb = DockerSandboxBackend(DockerSandboxConfig())
        container = AsyncMock()
        container._id = "c1"
        with patch.dict(os.environ, {"GITHUB_PROXY_URL": "http://proxy:8080"}):
            await sb._inject_secrets(container, github_token=None)

        container.put_archive.assert_called_once()
        call_args = container.put_archive.call_args
        tar_data = call_args[1]["data"]
        assert b"GITHUB_PROXY_URL" in tar_data
        assert b"http://proxy:8080" in tar_data


class TestDockerSandboxErrors:
    def test_execute_handles_exception(self, fake_run_async) -> None:
        fake_run_async.side_effect = RuntimeError("connection refused")
        sb = DockerSandbox("c1")

        result = sb.execute("echo hi")

        assert isinstance(result, ExecuteResponse)
        assert result.exit_code == -1
        assert "connection refused" in result.output


# ===========================================================================
# Factory registration integration
# ===========================================================================


def test_docker_registered_in_factories() -> None:
    from agent.utils.sandbox import SANDBOX_FACTORIES

    assert "docker" in SANDBOX_FACTORIES
    assert SANDBOX_FACTORIES["docker"] is create_docker_sandbox
