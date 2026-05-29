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

File operations
---------------
All six :class:`BackendProtocol` file operations (``ls``, ``read``,
``write``, ``edit``, ``grep``, ``glob``) are implemented directly on
:class:`DockerSandboxBackend` rather than inherited from
:class:`BaseSandbox`.  Key differences from the ``BaseSandbox``
execute-based defaults:

* **read / write / edit** — use Docker archive APIs
  (``get_archive`` / ``put_archive``) instead of execute-based Python
  scripts, avoiding large-file hangs and shell-quoting bugs.
* **write** — fails if the target file already exists (matching
  ``BaseSandbox._write_preflight`` behaviour).
* **grep** — uses ``grep -rHnFI`` (recursive, filename, line-number,
  fixed-strings, ignore-binary).  Output is parsed with
  ``rfind(':')``, so filenames containing colons are handled
  correctly; matched lines whose text contains colons may be
  silently dropped (opposite tradeoff to ``BaseSandbox``'s
  ``split(":", 2)``, which breaks on colon-containing paths).
* **ls** — uses ``ls -1F`` (``-F`` flag appends ``/`` to
  directories), lighter than ``BaseSandbox``'s Python
  ``os.scandir`` script.
* **glob** — uses a Python one-liner with ``sys.argv`` (no string
  interpolation), avoiding the RCE vector that affected earlier
  template-based approaches.
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import shlex
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
    EditResult,
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
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

    All six :class:`BackendProtocol` file operations are implemented
    directly (see module docstring for behavioural differences from
    :class:`BaseSandbox`).

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

        from agent.integrations.docker_sandbox import _aexecute

        try:
            out, exit_code, timed_out, truncated = run_async(
                _aexecute(self._container_id, command, exec_timeout),
                timeout=exec_timeout + 10,
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

        if timed_out and exit_code > 0:
            exit_code = -1

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

    # -- File operations (BackendProtocol) --------------------------------------

    def ls(self, path: str) -> LsResult:
        if self._closed or self._container_id is None:
            return LsResult(error="sandbox closed")

        quoted = shlex.quote(path)
        result = self.execute(f"ls -1F -- {quoted}")
        if result.exit_code != 0:
            err = result.output.strip()
            if "No such file" in err or "cannot access" in err:
                return LsResult(error=f"Path '{path}': path_not_found")
            return LsResult(error=err or f"ls failed (exit {result.exit_code})")

        entries: list[dict] = []
        for line in result.output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            is_dir = line.endswith("/")
            name = line[:-1] if is_dir else line
            full_path = os.path.join(path.rstrip("/"), name)
            entries.append({"path": full_path, "is_dir": is_dir})
        return LsResult(entries=entries)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        if self._closed or self._container_id is None:
            return ReadResult(error="sandbox closed")

        check = self.execute(f"test -d {shlex.quote(file_path)}")
        if check.exit_code == 0:
            return ReadResult(error=f"File '{file_path}': IsADirectoryError")

        responses = self.download_files([file_path])
        resp = responses[0]
        if resp.error:
            if resp.error == "file_not_found":
                return ReadResult(error=f"File '{file_path}': FileNotFoundError")
            return ReadResult(error=resp.error)

        raw = resp.content or b""
        content = raw.decode("utf-8", errors="replace")

        lines = content.split("\n")
        page = lines[offset : offset + limit]
        return ReadResult(file_data=FileData(content="\n".join(page), encoding="utf-8"))

    def _write_via_archive(self, file_path: str, content: str) -> WriteResult:
        parent = os.path.dirname(file_path)
        if parent:
            result = self.execute(f"mkdir -p {shlex.quote(parent)}")
            if result.exit_code != 0:
                return WriteResult(
                    error=f"Failed to create parent directory: {result.output.strip()}"
                )

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar_path = file_path.lstrip("/")
            info = tarfile.TarInfo(name=tar_path)
            encoded = content.encode("utf-8")
            info.size = len(encoded)
            info.mtime = int(time.time())
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(encoded))

        from agent.integrations.docker_sandbox import _put_archive_to_container

        try:
            run_async(_put_archive_to_container(self._container_id, buf.getvalue()))
        except Exception as exc:
            logger.exception("Error writing file %s", file_path)
            return WriteResult(error=f"Failed to write file '{file_path}': {exc}")

        return WriteResult(path=file_path)

    def write(self, file_path: str, content: str) -> WriteResult:
        if self._closed or self._container_id is None:
            return WriteResult(error="sandbox closed")

        # Fail if file already exists (matching BaseSandbox preflight behavior)
        exists = self.execute(f"test -f {shlex.quote(file_path)}")
        if exists.exit_code == 0:
            return WriteResult(error=f"File '{file_path}' already exists")

        return self._write_via_archive(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        if self._closed or self._container_id is None:
            return EditResult(error="sandbox closed")

        read_result = self.read(file_path)
        if read_result.error:
            return EditResult(error=read_result.error)

        content = read_result.file_data["content"]
        count = content.count(old_string)

        if count == 0:
            return EditResult(error=f"Error: String not found in file: '{old_string}'")
        if count > 1 and not replace_all:
            return EditResult(
                error=f"Error: String '{old_string}' appears multiple times. "
                "Use replace_all=True to replace all occurrences."
            )

        new_content = (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )
        write_result = self._write_via_archive(file_path, new_content)
        if write_result.error:
            return EditResult(error=write_result.error)

        return EditResult(path=file_path, occurrences=count)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        if self._closed or self._container_id is None:
            return GrepResult(error="sandbox closed")

        search_path = shlex.quote(path or ".")
        grep_opts = "-rHnFI"
        glob_flag = ""
        if glob:
            glob_flag = f"--include={shlex.quote(glob)}"
        pat = shlex.quote(pattern)

        cmd = f"grep {grep_opts} {glob_flag} -e {pat} {search_path}"
        result = self.execute(cmd)

        if result.exit_code >= 2:
            msg = result.output.strip() or f"grep exit code {result.exit_code}"
            return GrepResult(error=f"grep error: {msg}")

        output = result.output.rstrip()
        if not output:
            return GrepResult(matches=[])

        matches: list[dict] = []
        for line in output.split("\n"):
            if not line:
                continue
            last_colon = line.rfind(":")
            if last_colon == -1:
                continue
            text = line[last_colon + 1 :]
            rest = line[:last_colon]
            line_colon = rest.rfind(":")
            if line_colon == -1:
                continue
            try:
                line_num = int(rest[line_colon + 1 :])
            except ValueError:
                continue
            matches.append({"path": rest[:line_colon], "line": line_num, "text": text})

        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        if self._closed or self._container_id is None:
            return GlobResult(error="sandbox closed")

        script = (
            "import sys,glob,os,json;"
            "base=sys.argv[1];"
            "pat=sys.argv[2];"
            "if os.path.isdir(base):os.chdir(base);"
            "for f in glob.glob(pat,recursive=True):"
            "  print(json.dumps(dict(path=os.path.join(base,f),is_dir=os.path.isdir(f))))"
        )
        cmd = (
            f"python3 -c {shlex.quote(script)} {shlex.quote(str(path))} {shlex.quote(str(pattern))}"
        )
        result = self.execute(cmd)

        output = result.output.strip()
        if not output:
            return GlobResult(matches=[])

        matches: list[dict] = []
        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "error" in data:
                return GlobResult(error=f"Path '{path}': {data['error']}")
            matches.append({"path": data["path"], "is_dir": data["is_dir"]})

        return GlobResult(matches=matches)

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
