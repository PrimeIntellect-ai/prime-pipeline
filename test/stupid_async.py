import asyncio
from iroh_py import wait_for

async def wait_for_and_print(time: int):
    await wait_for(time)
    print(f"Waited for {time}")

async def sequential():
    print("Starting sequential")
    await wait_for_and_print(1)
    await wait_for_and_print(2)
    await wait_for_and_print(3)

async def concurrent():
    print("Starting concurrent")
    await asyncio.gather(
        wait_for_and_print(1),
        wait_for_and_print(2),
        wait_for_and_print(3)
    )

async def concurrent_with_tasks():
    print("Starting concurrent with tasks")
    tasks = [
        asyncio.create_task(wait_for_and_print(1)),
        asyncio.create_task(wait_for_and_print(2)),
        asyncio.create_task(wait_for_and_print(3))
    ]
    await asyncio.gather(*tasks)

asyncio.run(sequential())
asyncio.run(concurrent())
asyncio.run(concurrent_with_tasks())