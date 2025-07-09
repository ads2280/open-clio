from typing_extensions import TypeVar
import asyncio
from typing import Awaitable


ReturnValue = TypeVar("ReturnValue")


async def gated_coro(
    coro: Awaitable[ReturnValue], semaphore: asyncio.Semaphore
) -> ReturnValue:
    async with semaphore:
        return await coro
