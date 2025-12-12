import asyncio
from pyftg.socket.aio.gateway import Gateway
from kick_ai import KickAI
from display_info import DisplayInfo

async def main():
    gateway = Gateway(port=31415)
    gateway.register_ai("KickAI", KickAI())
    gateway.register_ai("DisplayInfo", DisplayInfo())
    await gateway.run_game(["ZEN", "ZEN"], ["KickAI", "DisplayInfo"], 1)
    await gateway.close()

asyncio.run(main())
