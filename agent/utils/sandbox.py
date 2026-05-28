import asyncio
import inspect
import logging
import os
import sys
import threading

from deepagents.backends.protocol import SandboxBackendProtocol

from agent.integrations.daytona import create_daytona_sandbox
from agent.integrations.docker_sandbox import _parse_mem, create_docker_sandbox
from agent.integrations.langsmith import create_langsmith_sandbox
from agent.integrations.local import create_local_sandbox
from agent.integrations.modal import create_modal_sandbox
from agent.integrations.runloop import create_runloop_sandbox

SANDBOX_FACTORIES = {
    "langsmith": create_langsmith_sandbox,
    "daytona": create_daytona_sandbox,
    "docker": create_docker_sandbox,
    "modal": create_modal_sandbox,
    "runloop": create_runloop_sandbox,
    "local": create_local_sandbox,
}

DEFAULT_MAX_CONCURRENT_CREATE = 4
SANDBOX_MAX_CONCURRENT_CREATE_ENV_VAR = "SANDBOX_MAX_CONCURRENT_CREATE"

_create_semaphore_lock = threading.Lock()
_create_semaphore: asyncio.Semaphore | None = None


async def _get_create_semaphore() -> asyncio.Semaphore:
    global _create_semaphore
    with _create_semaphore_lock:
        if _create_semaphore is None:
            limit = int(
                os.getenv(
                    SANDBOX_MAX_CONCURRENT_CREATE_ENV_VAR,
                    str(DEFAULT_MAX_CONCURRENT_CREATE),
                )
            )
            _create_semaphore = asyncio.Semaphore(limit)
        return _create_semaphore


async def create_sandbox(
    sandbox_id: str | None = None,
    github_token: str | None = None,
) -> SandboxBackendProtocol:
    """Create or reconnect to a sandbox using the configured provider.

    Supports both sync and async factories.  Detects the factory type at
    runtime via ``inspect.iscoroutinefunction``:
    - Async factory → awaited directly.
    - Sync factory → offloaded via ``asyncio.to_thread``.

    A global async semaphore limits concurrent sandbox creation across all
    providers.  The concurrency limit is read from
    ``SANDBOX_MAX_CONCURRENT_CREATE`` (default 4).

    The provider is selected via the SANDBOX_TYPE environment variable.
    Supported values: langsmith (default), daytona, docker, modal, runloop, local.

    Args:
        sandbox_id: Optional existing sandbox ID to reconnect to.
        github_token: Optional GitHub token passed through to factories
            that support it (e.g. langsmith proxy auth).

    Returns:
        A sandbox backend implementing SandboxBackendProtocol.
    """
    sem = await _get_create_semaphore()
    async with sem:
        sandbox_type = os.getenv("SANDBOX_TYPE", "langsmith")
        factory = SANDBOX_FACTORIES.get(sandbox_type)
        if not factory:
            supported = ", ".join(sorted(SANDBOX_FACTORIES))
            msg = f"Invalid sandbox type: {sandbox_type}. Supported types: {supported}"
            raise ValueError(msg)

        if inspect.iscoroutinefunction(factory):
            return await factory(sandbox_id=sandbox_id, github_token=github_token)

        accepts_token = "github_token" in inspect.signature(factory).parameters
        if accepts_token:
            return await asyncio.to_thread(factory, sandbox_id, github_token)
        return await asyncio.to_thread(factory, sandbox_id)


def _validate_sandbox_max_concurrent() -> None:
    raw = os.getenv(SANDBOX_MAX_CONCURRENT_CREATE_ENV_VAR, str(DEFAULT_MAX_CONCURRENT_CREATE))
    try:
        value = int(raw)
    except ValueError:
        msg = f"{SANDBOX_MAX_CONCURRENT_CREATE_ENV_VAR} must be an integer, got: {raw!r}"
        raise ValueError(msg) from None
    if value < 1:
        msg = f"{SANDBOX_MAX_CONCURRENT_CREATE_ENV_VAR} must be >= 1, got: {value}"
        raise ValueError(msg)


def _validate_docker_config() -> None:
    """Validate Docker sandbox configuration at startup.

    Checks:
    - Python version >= 3.12 (required for tarfile.extractall(filter='data'))
    - DOCKER_SANDBOX_IMAGE is set
    - All int fields parse as integers
    - DOCKER_HOST scheme validation (unix/ssh/tcp-with-tls only, unless insecure override)
    """
    if sys.version_info < (3, 12):
        raise RuntimeError(
            f"Docker sandbox requires Python 3.12+, got {sys.version_info.major}.{sys.version_info.minor}. "
            "tarfile.extractall(filter='data') is not available in earlier versions."
        )

    image = os.getenv("DOCKER_SANDBOX_IMAGE")
    if not image:
        raise ValueError("DOCKER_SANDBOX_IMAGE must be set when SANDBOX_TYPE=docker")

    int_fields = [
        "DOCKER_SANDBOX_CPU_LIMIT",
        "DOCKER_SANDBOX_PID_LIMIT",
        "DOCKER_SANDBOX_TIMEOUT",
        "DOCKER_SANDBOX_WALL_CLOCK_GRACE",
        "DOCKER_SANDBOX_MAX_CONCURRENT",
        "DOCKER_SANDBOX_MAX_OUTPUT_BYTES",
        "DOCKER_SANDBOX_CLEANUP_INTERVAL",
        "DOCKER_SANDBOX_ORPHAN_TTL",
    ]
    for field in int_fields:
        raw = os.getenv(field)
        if raw is None or raw == "":
            continue
        try:
            int(raw)
        except ValueError as exc:
            raise ValueError(f"{field} must be an integer, got: {raw!r}") from exc

    mem_limit = os.getenv("DOCKER_SANDBOX_MEM_LIMIT")
    if mem_limit:
        try:
            _parse_mem(mem_limit)
        except ValueError as exc:
            raise ValueError(
                f"DOCKER_SANDBOX_MEM_LIMIT must be an integer or suffixed value (e.g. 4g, 512m), got: {mem_limit!r}"
            ) from exc

    _validate_docker_host()


def _validate_docker_host() -> None:
    """Validate DOCKER_HOST scheme for security.

    Allowed schemes:
    - unix://... (local socket) → OK
    - ssh://... (SSH transport) → OK
    - tcp://... with TLS (DOCKER_TLS_VERIFY=1 + DOCKER_CERT_PATH) → OK
    - tcp://... without TLS → rejected unless DOCKER_SANDBOX_ALLOW_INSECURE_TCP=1 (logs WARN)
    - Other schemes → ValueError

    Raises ValueError for unsupported or insecure configurations.
    """
    logger = logging.getLogger("open_swe.docker")

    host = os.getenv("DOCKER_HOST", "")
    if not host:
        return

    if host.startswith("unix://") or host.startswith("ssh://"):
        return

    if host.startswith("tcp://"):
        tls_verify = os.getenv("DOCKER_TLS_VERIFY") == "1"
        cert_path = os.getenv("DOCKER_CERT_PATH")
        allow_insecure = os.getenv("DOCKER_SANDBOX_ALLOW_INSECURE_TCP") == "1"

        if tls_verify and cert_path:
            return

        if allow_insecure:
            logger.warning(
                "sandbox.insecure_tcp",
                extra={"docker_host": host},
            )
            return

        raise ValueError(
            "tcp:// DOCKER_HOST requires DOCKER_TLS_VERIFY=1 + DOCKER_CERT_PATH, "
            "or set DOCKER_SANDBOX_ALLOW_INSECURE_TCP=1 to override (not recommended)"
        )

    raise ValueError(f"Unsupported DOCKER_HOST scheme: {host}")


def validate_sandbox_startup_config() -> None:
    """Validate the configured sandbox provider's env vars at server startup.

    Raises ValueError if the active provider's configuration is invalid.
    Called from the FastAPI lifespan hook so errors surface at boot rather
    than on the first sandbox creation.
    """
    _validate_sandbox_max_concurrent()
    sandbox_type = os.getenv("SANDBOX_TYPE", "langsmith")
    if sandbox_type == "langsmith":
        from agent.integrations.langsmith import LangSmithProvider

        LangSmithProvider.validate_startup_config()
    elif sandbox_type == "docker":
        _validate_docker_config()
