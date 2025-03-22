import time
import asyncio

def test_async_sequential():
    async def sequential():
        await asyncio.sleep(1)
        await asyncio.sleep(2)
    t0 = time.time()
    asyncio.run(sequential())
    assert round(time.time() - t0) == 3

def test_async_concurrent():
    async def concurrent():
        await asyncio.gather(
            asyncio.sleep(1),
            asyncio.sleep(2)
        )
    
    t0 = time.time()
    asyncio.run(concurrent())
    assert round(time.time() - t0) == 2