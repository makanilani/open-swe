"""Docker sandbox backend implementation.

Implements :class:`SandboxBackendProtocol` using the shared Docker
client pool and background I/O loop.  All Docker daemon interaction
is delegated to async coroutines on the background event loop.

Lifecycle
---------
``start()`` → container created/reused → ``execute()`` /
``upload_files()`` / ``download_files()`` → ``close()``.

Reconnect-first: ``start()`` always checks for an existing managed
container (via the ``open-swe-managed`` label) before creating a new
one.  Secrets are injected via ``put_archive`` **before** the container
starts so the entrypoint script picks them up on first boot.
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import string
import tarfile
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiodocker import Docker
    from aiodocker.containers import DockerContainer

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    SandboxBackendProtocol,
)

from agent.integrations.docker_client import (
    _ensure_image,
    _get_docker_client,
    _release_docker_client,
)
from agent.integrations.docker_io_loop import run_async

logger = logging.getLogger("open_swe.docker_sandbox")

DEFAULT_IMAGE = "open-swe-sandbox:latest"
DEFAULT_TIMEOUT = 300

LABEL_SANDBOX = "open-swe-sandbox"
LABEL_MANAGED = "open-swe-managed"
LABEL_CREATED_AT = "open-swe-created-at"
LABEL_VERSION = "open-swe-version"

try:
    from importlib.metadata import version as _pkg_version

    _PKG_VERSION: str = _pkg_version("open-swe")
except Exception:
    _PKG_VERSION = "dev"

_MEM_SUFFIXES = {"k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12}


def _parse_mem(value: str) -> int:
    value = value.strip().lower()
    suffix = value[-1]
    if suffix in _MEM_SUFFIXES:
        return int(float(value[:-1]) * _MEM_SUFFIXES[suffix])
    return int(value)


@dataclass
class DockerSandboxConfig:
    """Configuration for :class:`DockerSandboxBackend`.

    Parameters are read from environment variables when not explicitly set.
    """

    image: str = field(default_factory=lambda: os.getenv("DOCKER_SANDBOX_IMAGE", DEFAULT_IMAGE))
    mem_limit: str | None = field(default_factory=lambda: os.getenv("DOCKER_SANDBOX_MEM_LIMIT"))
    cpu_limit: int | None = field(
        default_factory=lambda: (
            int(os.getenv("DOCKER_SANDBOX_CPU_LIMIT"))
            if os.getenv("DOCKER_SANDBOX_CPU_LIMIT")
            else None
        )
    )
    pids_limit: int | None = field(
        default_factory=lambda: (
            int(os.getenv("DOCKER_SANDBOX_PID_LIMIT"))
            if os.getenv("DOCKER_SANDBOX_PID_LIMIT")
            else None
        )
    )
    network: str | None = field(default_factory=lambda: os.getenv("DOCKER_SANDBOX_NETWORK_MODE"))
    exec_timeout: int = field(
        default_factory=lambda: int(os.getenv("DOCKER_SANDBOX_TIMEOUT", str(DEFAULT_TIMEOUT)))
    )


class DockerSandboxBackend(SandboxBackendProtocol):
    """Docker-container sandbox backend.

    Exposes a sync :class:`SandboxBackendProtocol` surface while delegating
    all Docker daemon I/O to async coroutines on a dedicated background
    event loop.

    .. admonition:: Reconnect-first
       :class: tip

       ``start()`` looks for an existing container with the
       ``open-swe-managed`` label before creating a new one.  This lets
       agent runs survive process restarts as long as the container is
       still alive.

    .. admonition:: Secrets injection
       :class: important

       Secrets (GitHub token, proxy URL) are written into
       ``/tmp/open-swe/secrets/`` via ``put_archive`` **before** the
       container is started, so the :file:`entrypoint.sh` script picks
       them up on first boot.
    """

    def __init__(self, config: DockerSandboxConfig | None = None) -> None:
        self._config = config or DockerSandboxConfig()
        self._container_id: str | None = None
        self._closed = False

    # -- SandboxBackendProtocol -------------------------------------------------

    @property
    def id(self) -> str:
        if self._container_id is None:
            return ""
        return self._container_id[:12]

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        if self._closed or self._container_id is None:
            return ExecuteResponse(output="", exit_code=1, truncated=False)

        exec_timeout = timeout or self._config.exec_timeout

        from agent.integrations.docker_sandbox import _exec_in_container

        try:
            out, exit_code = run_async(
                _exec_in_container(self._container_id, command, exec_timeout),
                timeout=exec_timeout,
            )
        except TimeoutError:
            return ExecuteResponse(
                output=f"Command timed out after {exec_timeout}s",
                exit_code=-1,
                truncated=False,
            )
        except Exception as exc:
            logger.exception("Unexpected error in execute")
            return ExecuteResponse(
                output=f"Error: {exc}",
                exit_code=-1,
                truncated=False,
            )

        truncated = len(out) > 500 * 1024
        return ExecuteResponse(output=out, exit_code=exit_code, truncated=truncated)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        if self._closed or self._container_id is None:
            return [FileUploadResponse(path=path, error="sandbox closed") for path, _ in files]

        from agent.integrations.docker_sandbox import _upload_files_to_container

        return run_async(_upload_files_to_container(self._container_id, files))

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        if self._closed or self._container_id is None:
            return [FileDownloadResponse(path=path, error="sandbox closed") for path in paths]

        from agent.integrations.docker_sandbox import _download_files_from_container

        return run_async(_download_files_from_container(self._container_id, paths))

    # -- Lifecycle --------------------------------------------------------------

    def start(self, github_token: str | None = None) -> DockerSandboxBackend:
        """Start the sandbox (reconnect-first).

        Reconnect-first semantics:
        1. Look for a managed container via the ``open-swe-managed`` label.
        2. If found, ensure it is running and bind to it.
        3. Otherwise create a new container, inject secrets **before**
           starting it, and wait for the health check.

        Args:
            github_token: Optional GitHub token written into
                ``/tmp/open-swe/secrets/GH_TOKEN`` before container start.

        Returns:
            ``self`` for chaining.
        """
        if self._container_id is not None:
            return self

        existing_id = run_async(self._try_reconnect())
        if existing_id:
            self._container_id = existing_id
            return self

        run_async(self._do_start_new(github_token))
        return self

    def close(self) -> None:
        """Stop and remove the underlying container (idempotent)."""
        if self._closed or self._container_id is None:
            return
        self._closed = True
        from agent.integrations.docker_sandbox import _stop_and_remove_container

        try:
            run_async(_stop_and_remove_container(self._container_id))
        except Exception:
            logger.exception("Error closing Docker sandbox %s", self._container_id)

    # -- Internal helpers -------------------------------------------------------

    async def _try_reconnect(self) -> str | None:
        """Find a managed container by label; start it if stopped.

        Returns the container ID on success, or ``None`` if no managed
        container exists (or the lookup fails).
        """
        from agent.integrations.docker_sandbox import _ensure_container_running

        try:
            client: Docker = await _get_docker_client()
            try:
                filters = json.dumps({"label": [f"{LABEL_MANAGED}=true"]})
                containers = await client.containers.list(all=True, filters=filters)
                if not containers:
                    return None
                containers.sort(
                    key=lambda c: c._container.get("Created", ""),
                    reverse=True,
                )
                cid = containers[0]._id
                await _ensure_container_running(cid)
                return cid
            finally:
                await _release_docker_client()
        except Exception:
            logger.exception("Reconnect failed, will create new container")
            return None

    async def _do_start_new(self, github_token: str | None = None) -> None:
        """Create a new container, inject secrets, start, and wait for health."""
        from agent.integrations.docker_sandbox import _wait_for_healthy

        client: Docker = await _get_docker_client()
        try:
            await _ensure_image(client, self._config.image)

            name = self._generate_container_name()
            config = self._build_container_config()
            container: DockerContainer = await client.containers.create(config, name=name)
            self._container_id = container._id
            logger.info("Created container %s (%s)", name, container._id)

            await self._inject_secrets(container, github_token)

            await container.start()
            logger.info("Started container %s", container._id)
        finally:
            await _release_docker_client()

        await _wait_for_healthy(self._container_id)

    def _generate_container_name(self) -> str:
        """Generate a unique container name.

        Pattern: ``open-swe-{short_id}-{uuid}`` where *short_id* is a
        random 8-character alphanumeric string.
        """
        short = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        uid = uuid.uuid4().hex
        return f"open-swe-{short}-{uid}"

    def _build_container_config(self) -> dict:
        """Build the Docker container configuration dict."""
        labels = {
            LABEL_SANDBOX: "true",
            LABEL_MANAGED: "true",
            LABEL_CREATED_AT: str(int(time.time())),
            LABEL_VERSION: _PKG_VERSION,
        }

        host_config: dict = {}
        if self._config.mem_limit:
            host_config["Memory"] = _parse_mem(self._config.mem_limit)
        if self._config.cpu_limit is not None:
            host_config["NanoCpus"] = self._config.cpu_limit
        if self._config.pids_limit is not None:
            host_config["PidsLimit"] = self._config.pids_limit
        if self._config.network:
            host_config["NetworkMode"] = self._config.network
        host_config["CapDrop"] = ["NET_RAW", "NET_ADMIN", "SYS_ADMIN"]
        host_config["ReadonlyRootfs"] = True
        host_config["SecurityOpt"] = ["no-new-privileges:true"]

        return {
            "Image": self._config.image,
            "Cmd": ["tail", "-f", "/dev/null"],
            "WorkingDir": "/workspace",
            "User": "swe-user",
            "Labels": labels,
            "HostConfig": host_config,
        }

    async def _inject_secrets(
        self,
        container: DockerContainer,
        github_token: str | None = None,
    ) -> None:
        """Inject secrets into the container **before** it starts.

        Writes files to ``/tmp/open-swe/secrets/`` via ``put_archive``.
        The container must be created but **not yet started** — the
        entrypoint script reads these files on first boot.
        """
        files: list[tuple[str, str]] = []
        if github_token:
            files.append(("GH_TOKEN", github_token))
        proxy_url = os.getenv("GITHUB_PROXY_URL")
        if proxy_url:
            files.append(("GITHUB_PROXY_URL", proxy_url))
        if not files:
            return

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for name, value in files:
                tar_path = f"tmp/open-swe/secrets/{name}"
                info = tarfile.TarInfo(name=tar_path)
                encoded = value.encode("utf-8")
                info.size = len(encoded)
                info.mtime = int(time.time())
                info.mode = 0o600
                tar.addfile(info, io.BytesIO(encoded))

        tar_data = buf.getvalue()
        await container.put_archive(path="/", data=tar_data)
        logger.info("Injected secrets into container %s", container._id)
