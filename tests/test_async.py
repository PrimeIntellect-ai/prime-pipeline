import asyncio
import time

import pytest


@pytest.mark.skip()
def test_async_sequential():
    async def sequential():
        await asyncio.sleep(0.1)
        await asyncio.sleep(0.2)

    t0 = time.time()
    asyncio.run(sequential())
    assert round(time.time() - t0, 1) == 0.3


@pytest.mark.skip()
def test_async_concurrent():
    async def concurrent():
        await asyncio.gather(asyncio.sleep(0.1), asyncio.sleep(0.2))

    t0 = time.time()
    asyncio.run(concurrent())
    assert round(time.time() - t0, 1) == 0.2
