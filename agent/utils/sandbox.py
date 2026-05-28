import asyncio
import inspect
import os
import threading

from deepagents.backends.protocol import SandboxBackendProtocol

from agent.integrations.daytona import create_daytona_sandbox
from agent.integrations.langsmith import create_langsmith_sandbox
from agent.integrations.local import create_local_sandbox
from agent.integrations.modal import create_modal_sandbox
from agent.integrations.runloop import create_runloop_sandbox

SANDBOX_FACTORIES = {
    "langsmith": create_langsmith_sandbox,
    "daytona": create_daytona_sandbox,
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
    Supported values: langsmith (default), daytona, modal, runloop, local.

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
