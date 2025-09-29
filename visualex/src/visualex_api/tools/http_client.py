import aiohttp
import asyncio

class HttpClient:
    _session: aiohttp.ClientSession = None

    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # Using a higher limit for a shared connector.
            # aiohttp's default is 100. Let's stick with that.
            connector = aiohttp.TCPConnector(ssl=False, limit=100)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close_session(self):
        if self._session and not self._session.closed:
            await self._session.close()
            # Recommended grace period for TCP connections to close
            await asyncio.sleep(0.250)
            self._session = None

# Global instance to be used across the application
http_client = HttpClient()
