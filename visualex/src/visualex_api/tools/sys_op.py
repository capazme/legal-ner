import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import structlog
from bs4 import BeautifulSoup
import requests
import aiohttp
from aiocache import Cache

from .http_client import http_client


class WebDriverManager:
    _driver = None

    def __init__(self):
        self.drivers = []
        log.info("WebDriverManager initialized")

    def get_driver(self, download_dir=None):
        if self._driver is None:
            if download_dir is None:
                download_dir = os.path.join(os.getcwd(), "download")
            log.info(f"Setting up WebDriver with download directory: {download_dir}")

            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920x1080")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")

            prefs = {
                "download.default_directory": download_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "plugins.always_open_pdf_externally": True
            }
            chrome_options.add_experimental_option("prefs", prefs)
            
            try:
                self._driver = webdriver.Chrome(options=chrome_options)
                log.info("WebDriver initialized successfully")
            except Exception as e:
                log.error(f"Failed to initialize WebDriver: {e}")
                raise
        return self._driver

    def close_drivers(self):
        """
        Closes all open WebDriver instances and clears the driver list.
        """
        log.info("Closing all WebDriver instances")
        if self._driver:
            try:
                self._driver.quit()
                log.info("WebDriver closed successfully")
            except Exception as e:
                log.warning(f"Failed to quit WebDriver: {e}")
            self._driver = None
        log.info("All WebDriver instances closed and cleared")

class BaseScraper:
    async def request_document(self, url):
        log.info(f"Consulting source - URL: {url}")
        session = await http_client.get_session()
        try:
            async with session.get(url, timeout=30) as response:
                response.raise_for_status()
                return await response.text()
        except aiohttp.ClientError as e:
            log.error(f"Error during consultation: {e}")
            raise ValueError(f"Problem with download: {e}")

    def parse_document(self, html_content):
        log.info("Parsing document content")
        return BeautifulSoup(html_content, 'html.parser')

log = structlog.get_logger()
web_driver_manager = WebDriverManager()
# Usage example:
# driver_manager = WebDriverManager()
# driver = driver_manager.setup_driver()
# driver_manager.close_drivers()
