import asyncio
import contextlib
import typing

# TODO: add class with mechanisms for staging inputs, running flow, and copying to dest


@contextlib.asynccontextmanager
async def get_prefect() -> typing.AsyncIterator[asyncio.subprocess.Process]:
    proc = await asyncio.create_subprocess_exec(
        *["prefect", "server", "start", "--no-ui"]
    )
    # sleep to ensure server started
    await asyncio.sleep(10)
    try:
        yield proc
    finally:
        if proc.returncode is None:
            proc.terminate()
