"""Tests for Docker sandbox startup configuration validation."""

from unittest.mock import patch

import pytest

from agent.utils.sandbox import (
    _validate_docker_config,
    _validate_docker_host,
    validate_sandbox_startup_config,
)


class _VersionInfo:
    """Mock version_info tuple with .major/.minor attributes."""

    def __init__(self, major: int, minor: int, micro: int = 0):
        self.major = major
        self.minor = minor
        self.micro = micro
        self._tuple = (major, minor, micro)

    def __getitem__(self, index: int) -> int:
        return self._tuple[index]

    def __lt__(self, other: tuple) -> bool:
        return self._tuple < other

    def __ge__(self, other: tuple) -> bool:
        return self._tuple >= other


class TestValidateDockerHost:
    """Tests for DOCKER_HOST scheme validation."""

    def test_unix_socket_accepted(self) -> None:
        """unix:// scheme should be accepted."""
        with patch.dict("os.environ", {"DOCKER_HOST": "unix:///var/run/docker.sock"}, clear=True):
            _validate_docker_host()

    def test_ssh_transport_accepted(self) -> None:
        """ssh:// scheme should be accepted."""
        with patch.dict("os.environ", {"DOCKER_HOST": "ssh://user@host"}, clear=True):
            _validate_docker_host()

    def test_tcp_with_tls_accepted(self) -> None:
        """tcp:// with TLS should be accepted."""
        with patch.dict(
            "os.environ",
            {
                "DOCKER_HOST": "tcp://host:2376",
                "DOCKER_TLS_VERIFY": "1",
                "DOCKER_CERT_PATH": "/path/to/certs",
            },
            clear=True,
        ):
            _validate_docker_host()

    def test_tcp_without_tls_rejected(self) -> None:
        """tcp:// without TLS should be rejected."""
        with patch.dict(
            "os.environ",
            {"DOCKER_HOST": "tcp://host:2375"},
            clear=True,
        ):
            with pytest.raises(ValueError, match="DOCKER_TLS_VERIFY"):
                _validate_docker_host()

    def test_tcp_insecure_override_allowed_with_warning(self, caplog) -> None:
        """tcp:// without TLS should be allowed with insecure override."""
        with patch.dict(
            "os.environ",
            {
                "DOCKER_HOST": "tcp://host:2375",
                "DOCKER_SANDBOX_ALLOW_INSECURE_TCP": "1",
            },
            clear=True,
        ):
            _validate_docker_host()
            assert "sandbox.insecure_tcp" in caplog.text

    def test_unsupported_scheme_rejected(self) -> None:
        """Unsupported schemes should be rejected."""
        with patch.dict("os.environ", {"DOCKER_HOST": "http://host:2375"}, clear=True):
            with pytest.raises(ValueError, match="Unsupported DOCKER_HOST scheme"):
                _validate_docker_host()

    def test_empty_docker_host_accepted(self) -> None:
        """Empty DOCKER_HOST should be accepted (uses default local socket)."""
        with patch.dict("os.environ", {"DOCKER_HOST": ""}, clear=True):
            _validate_docker_host()

    def test_no_docker_host_env_accepted(self) -> None:
        """Missing DOCKER_HOST should be accepted (uses default local socket)."""
        with patch.dict("os.environ", {}, clear=True):
            _validate_docker_host()


class TestValidateDockerConfig:
    """Tests for Docker sandbox configuration validation."""

    def test_python_version_check_passes(self) -> None:
        """Python 3.12+ should pass validation."""
        with patch("agent.utils.sandbox.sys") as mock_sys:
            mock_sys.version_info = _VersionInfo(3, 12, 0)
            with patch.dict(
                "os.environ",
                {"DOCKER_SANDBOX_IMAGE": "open-swe-sandbox:latest"},
                clear=True,
            ):
                _validate_docker_config()


class TestValidateSandboxStartupConfig:
    """Tests for the top-level validate_sandbox_startup_config dispatch."""

    def test_docker_dispatch_passes_with_valid_config(self) -> None:
        """Docker validation should pass with valid config."""
        with patch("agent.utils.sandbox.sys") as mock_sys:
            mock_sys.version_info = _VersionInfo(3, 12, 0)
            with patch.dict(
                "os.environ",
                {
                    "DOCKER_SANDBOX_IMAGE": "open-swe-sandbox:latest",
                    "SANDBOX_TYPE": "docker",
                },
                clear=True,
            ):
                validate_sandbox_startup_config()

    def test_docker_dispatch_rejects_missing_image(self) -> None:
        """Docker validation should reject missing DOCKER_SANDBOX_IMAGE."""
        with patch("agent.utils.sandbox.sys") as mock_sys:
            mock_sys.version_info = _VersionInfo(3, 12, 0)
            with patch.dict(
                "os.environ",
                {"SANDBOX_TYPE": "docker"},
                clear=True,
            ):
                with pytest.raises(ValueError, match="DOCKER_SANDBOX_IMAGE must be set"):
                    validate_sandbox_startup_config()

    def test_langsmith_dispatch_does_not_call_docker(self) -> None:
        """LangSmith validation should not call Docker validation."""
        with patch.dict(
            "os.environ",
            {"SANDBOX_TYPE": "langsmith", "DEFAULT_SANDBOX_SNAPSHOT_ID": "snap-1"},
            clear=True,
        ):
            validate_sandbox_startup_config()

    def test_default_sandbox_type_langsmith(self) -> None:
        """Default SANDBOX_TYPE should be langsmith (no Docker validation)."""
        with patch.dict(
            "os.environ",
            {"DEFAULT_SANDBOX_SNAPSHOT_ID": "snap-1"},
            clear=True,
        ):
            validate_sandbox_startup_config()

    def test_python_version_check_fails(self) -> None:
        """Python < 3.12 should fail validation."""
        with patch("agent.utils.sandbox.sys") as mock_sys:
            mock_sys.version_info = _VersionInfo(3, 11, 0)
            with patch.dict(
                "os.environ",
                {"DOCKER_SANDBOX_IMAGE": "open-swe-sandbox:latest"},
                clear=True,
            ):
                with pytest.raises(RuntimeError, match="Python 3.12"):
                    _validate_docker_config()

    def test_missing_image_rejected(self) -> None:
        """Missing DOCKER_SANDBOX_IMAGE should be rejected."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="DOCKER_SANDBOX_IMAGE must be set"):
                _validate_docker_config()

    def test_empty_image_rejected(self) -> None:
        """Empty DOCKER_SANDBOX_IMAGE should be rejected."""
        with patch.dict("os.environ", {"DOCKER_SANDBOX_IMAGE": ""}, clear=True):
            with pytest.raises(ValueError, match="DOCKER_SANDBOX_IMAGE must be set"):
                _validate_docker_config()

    def test_valid_int_fields_accepted(self) -> None:
        """Valid integer fields should be accepted."""
        with patch.dict(
            "os.environ",
            {
                "DOCKER_SANDBOX_IMAGE": "open-swe-sandbox:latest",
                "DOCKER_SANDBOX_CPU_LIMIT": "2000000000",
                "DOCKER_SANDBOX_MEM_LIMIT": "4294967296",
                "DOCKER_SANDBOX_PID_LIMIT": "100",
                "DOCKER_SANDBOX_TIMEOUT": "300",
            },
            clear=True,
        ):
            _validate_docker_config()

    def test_invalid_int_field_rejected(self) -> None:
        """Invalid integer fields should be rejected."""
        with patch.dict(
            "os.environ",
            {
                "DOCKER_SANDBOX_IMAGE": "open-swe-sandbox:latest",
                "DOCKER_SANDBOX_TIMEOUT": "not-a-number",
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="DOCKER_SANDBOX_TIMEOUT must be an integer"):
                _validate_docker_config()

    def test_optional_int_fields_can_be_unset(self) -> None:
        """Optional integer fields can be unset."""
        with patch.dict(
            "os.environ",
            {"DOCKER_SANDBOX_IMAGE": "open-swe-sandbox:latest"},
            clear=True,
        ):
            _validate_docker_config()

    def test_optional_int_fields_can_be_empty(self) -> None:
        """Optional integer fields can be empty strings."""
        with patch.dict(
            "os.environ",
            {
                "DOCKER_SANDBOX_IMAGE": "open-swe-sandbox:latest",
                "DOCKER_SANDBOX_CPU_LIMIT": "",
                "DOCKER_SANDBOX_MEM_LIMIT": "",
            },
            clear=True,
        ):
            _validate_docker_config()

    def test_full_valid_config(self) -> None:
        """Full valid configuration should pass."""
        with patch.dict(
            "os.environ",
            {
                "DOCKER_SANDBOX_IMAGE": "open-swe-sandbox:latest",
                "DOCKER_SANDBOX_CPU_LIMIT": "2000000000",
                "DOCKER_SANDBOX_MEM_LIMIT": "4294967296",
                "DOCKER_SANDBOX_PID_LIMIT": "100",
                "DOCKER_SANDBOX_TIMEOUT": "300",
                "DOCKER_SANDBOX_WALL_CLOCK_GRACE": "10",
                "DOCKER_SANDBOX_MAX_CONCURRENT": "4",
                "DOCKER_SANDBOX_MAX_OUTPUT_BYTES": "5242880",
                "DOCKER_SANDBOX_CLEANUP_INTERVAL": "300",
                "DOCKER_SANDBOX_ORPHAN_TTL": "86400",
                "DOCKER_HOST": "unix:///var/run/docker.sock",
            },
            clear=True,
        ):
            _validate_docker_config()
