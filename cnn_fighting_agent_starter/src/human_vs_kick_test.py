import asyncio
import os

from pyftg.socket.aio.gateway import Gateway
from human_keyboard import HumanKeyboardAgent
from kick_ai import KickAI


async def main():
    port = int(os.environ.get("PORT", "31415"))
    print(f"[DEBUG] Connecting to 127.0.0.1:{port}")

    g = Gateway(host="127.0.0.1", port=port)
    g.register_ai("HUMAN", HumanKeyboardAgent())
    g.register_ai("KICK", KickAI())

    try:
        await g.run_game(["ZEN", "ZEN"], ["HUMAN", "KICK"], 1)
    finally:
        await g.close()


if __name__ == "__main__":
    asyncio.run(main())
