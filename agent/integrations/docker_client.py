"""Shared async Docker client pool with ref counting and atexit cleanup."""

import atexit
import logging
import os
import threading

import aiodocker

from agent.integrations.docker_io_loop import run_async

logger = logging.getLogger("open_swe.docker")

_client: aiodocker.Docker | None = None
_client_ref_count = 0
_client_lock = threading.Lock()


async def _get_docker_client() -> aiodocker.Docker:
    global _client, _client_ref_count

    with _client_lock:
        if _client is None:
            host = os.getenv("DOCKER_HOST")
            _client = aiodocker.Docker(url=host) if host else aiodocker.Docker()
            logger.info("Initialized shared aiodocker client")

        _client_ref_count += 1
        return _client


async def _release_docker_client() -> None:
    global _client, _client_ref_count

    with _client_lock:
        _client_ref_count -= 1

        if _client_ref_count <= 0 and _client is not None:
            try:
                await _client.close()
            except Exception:
                logger.exception("Error closing shared aiodocker client")
            finally:
                _client = None
                _client_ref_count = 0
                logger.info("Closed shared aiodocker client")


async def _force_close_client() -> None:
    global _client, _client_ref_count

    with _client_lock:
        if _client is not None:
            try:
                await _client.close()
            except Exception:
                logger.exception("Error during force-close of aiodocker client")
            finally:
                _client = None
                _client_ref_count = 0


def _atexit_close() -> None:
    if _client is None:
        return
    try:
        run_async(_force_close_client())
    except Exception:
        logger.exception("atexit close failed")


atexit.register(_atexit_close)
