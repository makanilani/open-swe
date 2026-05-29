"""Docker sandbox backend implementation.

Builds, creates, starts, and manages Docker containers as sandbox
backends.  All Docker daemon interaction runs on a shared background
event loop via ``docker_io_loop.run_async`` so that the sync
``SandboxBackendProtocol`` methods work without blocking the caller.
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
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    SandboxBackendProtocol,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from aiodocker import Docker
    from aiodocker.containers import DockerContainer

from agent.integrations.docker_client import (
    _get_docker_client,
    _release_docker_client,
)
from agent.integrations.docker_io_loop import run_async

logger = logging.getLogger("open_swe.docker_sandbox")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_TAG = "open-swe-sandbox:latest"
DEFAULT_CONTAINER_PREFIX = "open-swe-"
DEFAULT_EXEC_TIMEOUT = 300
IMAGE_BUILD_TIMEOUT = 300  # seconds

HEALTH_CHECK_INTERVAL = 0.5
HEALTH_CHECK_MAX_RETRIES = 60  # ~30 seconds total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_suffix(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _build_image_name() -> str:
    return os.getenv("DOCKER_SANDBOX_IMAGE", DEFAULT_IMAGE_TAG)


def _build_container_name() -> str:
    prefix = os.getenv("DOCKER_CONTAINER_PREFIX", DEFAULT_CONTAINER_PREFIX)
    return f"{prefix}{_random_suffix()}"


# -- async Docker helpers (called via run_async) ----------------------------


async def _ensure_image_exists(image_name: str) -> None:
    client: Docker = await _get_docker_client()
    try:
        try:
            await client.images.inspect(image_name)
            return
        except Exception:
            pass

        # Image not found locally — try to build from the bundled Dockerfile.
        docker_dir = os.path.join(os.path.dirname(__file__), "docker")
        if not os.path.isdir(docker_dir):
            raise RuntimeError(f"Docker build context not found: {docker_dir}")

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for name in ("Dockerfile", "entrypoint.sh"):
                fpath = os.path.join(docker_dir, name)
                info = tarfile.TarInfo(name=name)
                info.size = os.path.getsize(fpath)
                info.mtime = int(os.path.getmtime(fpath))
                with open(fpath, "rb") as f:
                    tar.addfile(info, f)
        buf.seek(0)
        raw_tar = buf.read()

        logger.info("Building Docker image %s …", image_name)
        await client.images.build(
            fileobj=raw_tar,
            tag=image_name,
            encoding="gzip",
            rm=True,
            timeout=IMAGE_BUILD_TIMEOUT,
        )
        logger.info("Docker image %s built successfully", image_name)
    finally:
        await _release_docker_client()


async def _create_and_start_container(
    container_name: str,
    image_name: str,
    gh_token: str | None = None,
    github_proxy_url: str | None = None,
) -> str:
    client: Docker = await _get_docker_client()
    try:
        env: list[str] = []
        if gh_token:
            env.append(f"GH_TOKEN={gh_token}")
        if github_proxy_url:
            env.append(f"GITHUB_PROXY_URL={github_proxy_url}")

        host_config: dict = {}
        mem_limit = os.getenv("DOCKER_SANDBOX_MEM_LIMIT")
        if mem_limit:
            host_config["Memory"] = _parse_mem(mem_limit)
        cpu_limit = os.getenv("DOCKER_SANDBOX_CPU_LIMIT")
        if cpu_limit:
            host_config["NanoCpus"] = int(cpu_limit)
        network = os.getenv("DOCKER_SANDBOX_NETWORK_MODE")
        if network:
            host_config["NetworkMode"] = network

        config: dict = {
            "Image": image_name,
            "Cmd": ["tail", "-f", "/dev/null"],
            "Env": env if env else None,
            "WorkingDir": "/workspace",
            "User": "swe-user",
            "Labels": {
                "open-swe-sandbox": "true",
                "open-swe-managed": "true",
                "open-swe-created-at": str(int(time.time())),
            },
            "HostConfig": host_config or None,
        }

        container: DockerContainer = await client.containers.run(config, name=container_name)
        logger.info("Container %s (%s) started", container_name, container._id)

        return container._id
    finally:
        await _release_docker_client()


async def _find_existing_container() -> str | None:
    """Find a running container managed by open-swe via label.

    Returns the container ID of the most recently created matching
    container, or ``None`` if none is found.
    """
    client: Docker = await _get_docker_client()
    try:
        filters = json.dumps({"label": ["open-swe-sandbox=true"]})
        containers = await client.containers.list(all=True, filters=filters)
        if not containers:
            return None
        # Prefer the most recently created container.
        containers.sort(
            key=lambda c: c._container.get("Created", ""),
            reverse=True,
        )
        return containers[0]._id
    except Exception:
        logger.exception("Error finding existing container")
        return None
    finally:
        await _release_docker_client()


async def _ensure_container_running(container_id: str) -> None:
    client: Docker = await _get_docker_client()
    try:
        container: DockerContainer = await client.containers.get(container_id)
        info = await container.show()
        status = (info.get("State") or {}).get("Status", "")
        if status != "running":
            logger.info("Starting stopped container %s (status=%s)", container_id, status)
            await container.start()
    finally:
        await _release_docker_client()


async def _stop_and_remove_container(container_id: str) -> None:
    client: Docker = await _get_docker_client()
    try:
        container: DockerContainer = await client.containers.get(container_id)
        try:
            await container.stop(t=5)
        except Exception:
            pass
        try:
            await container.delete(force=True)
        except Exception:
            pass
        logger.info("Container %s stopped and removed", container_id)
    finally:
        await _release_docker_client()


async def _exec_in_container(
    container_id: str,
    command: str,
    timeout: int,
) -> tuple[str, int]:
    client: Docker = await _get_docker_client()
    try:
        container: DockerContainer = await client.containers.get(container_id)
        exec_obj = await container.exec(
            cmd=["/bin/sh", "-c", command],
            stdout=True,
            stderr=True,
        )
        raw: bytes = await exec_obj.start(detach=False)
        out = raw.decode("utf-8", errors="replace")
        inspect_data = await exec_obj.inspect()
        exit_code: int = inspect_data.get("ExitCode", -1)
        return out, exit_code
    finally:
        await _release_docker_client()


async def _upload_files_to_container(
    container_id: str,
    files: list[tuple[str, bytes]],
) -> list[FileUploadResponse]:
    client: Docker = await _get_docker_client()
    try:
        container: DockerContainer = await client.containers.get(container_id)
        responses: list[FileUploadResponse] = []

        # Build a single tar with all files (paths relative to /)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for path, content in files:
                try:
                    tar_path = path.lstrip("/")
                    info = tarfile.TarInfo(name=tar_path)
                    info.size = len(content)
                    info.mtime = int(time.time())
                    tar.addfile(info, io.BytesIO(content))
                except Exception as exc:
                    responses.append(FileUploadResponse(path=path, error=str(exc)))
        tar_data = buf.getvalue()

        if responses and not any(r.error is None for r in responses):
            # All files failed during tar creation — nothing to upload.
            for path, _ in files:
                if not any(r.path == path for r in responses):
                    responses.append(FileUploadResponse(path=path, error="tar creation failed"))
            return responses

        # Upload the tar archive to the container root.
        try:
            await container.put_archive(path="/", data=tar_data)
        except Exception as exc:
            err = str(exc)
            for path, _ in files:
                if not any(r.path == path for r in responses):
                    responses.append(FileUploadResponse(path=path, error=err))
            return responses

        # Mark any remaining files as successful.
        for path, _ in files:
            if not any(r.path == path for r in responses):
                responses.append(FileUploadResponse(path=path))

        return responses
    finally:
        await _release_docker_client()


async def _download_files_from_container(
    container_id: str,
    paths: list[str],
) -> list[FileDownloadResponse]:
    client: Docker = await _get_docker_client()
    try:
        container: DockerContainer = await client.containers.get(container_id)
        responses: list[FileDownloadResponse] = []

        for path in paths:
            try:
                tar_file: tarfile.TarFile = await container.get_archive(path)
                extracted: dict[str, bytes] = {}
                for member in tar_file.getmembers():
                    if member.isfile():
                        f = tar_file.extractfile(member)
                        if f is not None:
                            extracted[member.name] = f.read()
                # get_archive returns tar entries whose names are relative to path.
                # If we got exactly one member matching the basename, use it;
                # otherwise return the first file.
                target_name = path.rstrip("/").rsplit("/", 1)[-1]
                content: bytes | None = None
                for name, data in extracted.items():
                    if name == target_name or name == path.lstrip("/"):
                        content = data
                        break
                if content is None and extracted:
                    content = next(iter(extracted.values()))

                if content is not None:
                    responses.append(FileDownloadResponse(path=path, content=content))
                else:
                    responses.append(FileDownloadResponse(path=path, error="file_not_found"))
            except Exception as exc:
                responses.append(FileDownloadResponse(path=path, error=str(exc)))

        return responses
    finally:
        await _release_docker_client()


async def _put_archive_to_container(container_id: str, tar_data: bytes) -> None:
    """Write a tar archive into the container at root path via put_archive."""
    client: Docker = await _get_docker_client()
    try:
        container: DockerContainer = await client.containers.get(container_id)
        await container.put_archive(path="/", data=tar_data)
    finally:
        await _release_docker_client()


async def _wait_for_healthy(container_id: str) -> None:
    for _attempt in range(1, HEALTH_CHECK_MAX_RETRIES + 1):
        out, _ = await _exec_in_container(
            container_id,
            "test -f /tmp/open-swe/ready",
            timeout=5,
        )
        if not out:
            return  # healthy
        await _check_sleep(HEALTH_CHECK_INTERVAL)
    raise TimeoutError(
        f"Container {container_id} did not become healthy "
        f"within {HEALTH_CHECK_MAX_RETRIES * HEALTH_CHECK_INTERVAL}s"
    )


async def _check_sleep(interval: float) -> None:
    """Async sleep bridge (runs via run_async → background loop)."""
    await __import__("asyncio").sleep(interval)


# -- memory parsing ---------------------------------------------------------

_MEM_SUFFIXES = {"k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12}


def _parse_mem(value: str) -> int:
    value = value.strip().lower()
    suffix = value[-1]
    if suffix in _MEM_SUFFIXES:
        return int(float(value[:-1]) * _MEM_SUFFIXES[suffix])
    return int(value)


# ===========================================================================
# DockerSandbox
# ===========================================================================


class DockerSandbox(BaseSandbox):
    """Docker-container-based sandbox implementing
    :class:`SandboxBackendProtocol`.

    All file operations (``ls``, ``read``, ``write``, ``edit``, ``grep``,
    ``glob``) are inherited from :class:`BaseSandbox` and work by calling
    :meth:`execute`.  Only the three core primitives are implemented here.
    """

    def __init__(self, container_id: str) -> None:
        self._container_id = container_id
        self._closed = False

    # -- SandboxBackendProtocol ------------------------------------------------

    @property
    def id(self) -> str:
        return self._container_id[:12]

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        if self._closed:
            return ExecuteResponse(output="", exit_code=1, truncated=False)

        exec_timeout = timeout or int(
            os.getenv("DOCKER_SANDBOX_TIMEOUT", str(DEFAULT_EXEC_TIMEOUT))
        )

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
        return run_async(_upload_files_to_container(self._container_id, files))

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return run_async(_download_files_from_container(self._container_id, paths))

    # -- lifecycle (not part of the protocol) ---------------------------------

    @classmethod
    def start(
        cls,
        gh_token: str | None = None,
        github_proxy_url: str | None = None,
    ) -> DockerSandbox:
        """Get-or-create a Docker sandbox.

        Reconnect-first: if a container with the ``open-swe-sandbox`` label
        already exists it is reused (started if stopped).  Otherwise a new
        container is created and waited on for the health check.
        """
        image = _build_image_name()
        run_async(_ensure_image_exists(image))

        existing_id = run_async(_find_existing_container())
        if existing_id:
            logger.info("Reusing existing container %s", existing_id)
            run_async(_ensure_container_running(existing_id))
            sandbox = cls(existing_id)
        else:
            name = _build_container_name()
            cid = run_async(_create_and_start_container(name, image, gh_token, github_proxy_url))
            sandbox = cls(cid)
            run_async(_wait_for_healthy(cid))

        if gh_token:
            sandbox._configure_git_credentials(gh_token)

        return sandbox

    @classmethod
    def reconnect(cls, container_id: str) -> DockerSandbox:
        """Reconnect to an existing Docker container (start if stopped)."""
        run_async(_ensure_container_running(container_id))
        return cls(container_id)

    def close(self) -> None:
        """Stop and remove the underlying container.

        This method is **not** part of the sandbox protocol — callers
        must invoke it explicitly when they want to tear down the
        container.
        """
        if self._closed:
            return
        self._closed = True
        try:
            run_async(_stop_and_remove_container(self._container_id))
        except Exception:
            logger.exception("Error closing Docker sandbox %s", self._container_id)

    # -- internal helpers -----------------------------------------------------

    def _configure_git_credentials(self, token: str) -> None:
        """Set up git credential store inside the container."""
        self.execute(
            "git config --global credential.helper store && "
            f"echo 'https://x-access-token:{token}@github.com' > ~/.git-credentials && "
            "chmod 600 ~/.git-credentials"
        )


# ===========================================================================
# Factory
# ===========================================================================


async def create_docker_sandbox(
    sandbox_id: str | None = None,
    github_token: str | None = None,
) -> SandboxBackendProtocol:
    """Create or reconnect to a Docker sandbox.

    Args:
        sandbox_id: Existing container ID to reconnect to.
            When ``None`` a new container is created.
        github_token: GitHub token for git-credential injection
            inside the container (only used for new containers).

    Returns:
        A :class:`DockerSandboxBackend` instance.
    """
    from agent.integrations.docker import DockerSandboxBackend, DockerSandboxConfig

    config = DockerSandboxConfig()
    backend = DockerSandboxBackend(config)

    if sandbox_id:
        backend._container_id = sandbox_id
        run_async(_ensure_container_running(sandbox_id))
    else:
        backend.start(github_token=github_token)

    return backend
