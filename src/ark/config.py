import os
from pathlib import Path

import aiohttp


def get_url() -> str:
    return os.environ.get("ARK_URL", "http://localhost:7070")


def get_home() -> Path:
    return Path(os.environ.get("ARK_HOME", str(Path.home() / ".tinyclaw")))


async def post(path: str, payload: dict) -> dict:
    url = get_url().rstrip("/") + path
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()
