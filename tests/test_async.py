import time
import asyncio
from iroh_py import wait_for

def test_async_sequential():
    async def sequential():
        await wait_for(1)
        await wait_for(2)
    t0 = time.time()
    asyncio.run(sequential())
    assert round(time.time() - t0) == 3

def test_async_concurrent():
    async def concurrent():
        await asyncio.gather(
            wait_for(1),
            wait_for(2)
        )
    
    t0 = time.time()
    asyncio.run(concurrent())
    assert round(time.time() - t0) == 2