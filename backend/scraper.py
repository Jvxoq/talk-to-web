import asyncio
import re
import aiohttp

from bs4 import BeautifulSoup
from loguru import logger

# Extract url from user prompt
def extract_urls(text: str) -> list[str]:
    url_pattern = r"(?P<url>https?:\/\/[^\s]+)"
    urls = re.findall(url_pattern, text)
    return urls

# Parse inner text from html
def parse_inner_text(html_string: str) -> str:
    soup = BeautifulSoup(html_string, "lxml")
    if content := soup.find("div", id="bodyContent"):
        return content.get_text()
    logger.warning("Could not parse html content")
    return ""

# Fetch html content of url
async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as response:
        html_string = await response.text()
        return parse_inner_text(html_string)

# Fetch html content from urls
async def fetch_all(urls: list[str]) -> str:
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[fetch(session, url) for url in urls],
            return_exceptions=True,
        )
    success_results = [result for result in results if isinstance(result, str)]
    if len(success_results) != len(results):
        logger.warning("Some urls couldn't be fetched")
    return " ".join(success_results)
