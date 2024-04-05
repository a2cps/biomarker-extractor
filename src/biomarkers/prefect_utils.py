import asyncio
import contextlib
from asyncio import subprocess
from collections import abc

# TODO: add class with mechanisms for staging inputs, running flow, and copying to dest


async def _startup() -> subprocess.Process:
    proc = await asyncio.create_subprocess_exec(
        *["prefect", "server", "start", "--no-ui"]
    )

    # sleep to ensure server started
    await asyncio.sleep(10)

    return proc


@contextlib.contextmanager
def get_prefect() -> abc.Generator[None, None, None]:
    proc = asyncio.run(_startup())
    try:
        yield
    finally:
        proc.terminate()
