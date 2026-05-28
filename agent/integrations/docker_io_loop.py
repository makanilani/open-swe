"""Shared background asyncio event loop for Docker sandbox I/O.

All Docker daemon interaction (container create, exec, file ops) runs
on a single background event loop in a daemon thread.  Sync callers
use ``run_async()`` which does a threadsafe submit + blocking wait.
"""

import asyncio
import threading

_io_lock: threading.Lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None


def _start_io_loop() -> None:
    """Lazy-init the background daemon thread + event loop (idempotent)."""
    global _loop, _thread

    with _io_lock:
        if _loop is not None:
            return
        _loop = asyncio.new_event_loop()
        _thread = threading.Thread(
            target=_loop.run_forever,
            daemon=True,
            name="docker-io-loop",
        )
        _thread.start()


def run_async(coro, *, timeout: float | None = None):
    """Submit a coroutine to the background loop and block for the result.

    Args:
        coro: The coroutine to execute on the I/O loop.
        timeout: Optional timeout in seconds for the blocking wait.

    Returns:
        The return value of *coro*.

    Raises:
        Any exception raised inside *coro*, or
        concurrent.futures.TimeoutError on timeout.
    """
    _start_io_loop()
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=timeout)
