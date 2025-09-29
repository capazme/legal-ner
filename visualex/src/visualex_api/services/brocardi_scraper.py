import structlog
import re
import os
from typing import Optional, Tuple, Union, Dict, Any, List
from dataclasses import dataclass

import aiohttp
import asyncio
import requests
from bs4 import BeautifulSoup
from aiocache import cached, Cache
from aiocache.serializers import JsonSerializer

from ..tools.map import BROCARDI_CODICI
from ..tools.norma import NormaVisitata
from ..tools.text_op import normalize_act_type
from ..tools.sys_op import BaseScraper
from ..tools.http_client import http_client

# Configurazione del logger di modulo
log = structlog.get_logger()


# Costanti per la configurazione
BASE_URL: str = "https://brocardi.it"
MAX_CONCURRENT_REQUESTS = 3  # Limitiamo le richieste concorrenti
REQUEST_TIMEOUT = 10  # Timeout per le richieste HTTP
RETRY_ATTEMPTS = 2  # Numero di retry per richiesta fallita
RETRY_DELAY = 1.0  # Delay tra i retry in secondi


@dataclass
class RequestConfig:
    """Configurazione per le richieste HTTP"""
    timeout: int = REQUEST_TIMEOUT
    retry_attempts: int = RETRY_ATTEMPTS
    retry_delay: float = RETRY_DELAY


class BrocardiScraper(BaseScraper):
    def __init__(self) -> None:
        log.info("Initializing BrocardiScraper")
        self.knowledge: List[Dict[str, Any]] = [BROCARDI_CODICI]
        self.request_config = RequestConfig()
        # Semaforo per limitare le richieste concorrenti
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def _make_request_with_retry(self, session: aiohttp.ClientSession, url: str, 
                                       config: RequestConfig = None) -> Optional[str]:
        """
        Effettua una richiesta HTTP con retry automatico e gestione errori migliorata.
        """
        if config is None:
            config = self.request_config
            
        for attempt in range(config.retry_attempts + 1):
            try:
                # Acquisisce il semaforo per limitare le richieste concorrenti
                async with self.semaphore:
                    timeout = aiohttp.ClientTimeout(total=config.timeout)
                    log.debug(f"Attempting request {attempt + 1}/{config.retry_attempts + 1} to: {url}")
                    
                    async with session.get(url, timeout=timeout) as response:
                        response.raise_for_status()
                        return await response.text()
                        
            except asyncio.TimeoutError:
                log.warning(f"Timeout for URL {url} on attempt {attempt + 1}")
            except aiohttp.ClientError as e:
                log.warning(f"HTTP error for URL {url} on attempt {attempt + 1}: {e}")
            except Exception as e:
                log.error(f"Unexpected error for URL {url} on attempt {attempt + 1}: {e}")
            
            # Se non è l'ultimo tentativo, aspetta prima di riprovare
            if attempt < config.retry_attempts:
                wait_time = config.retry_delay * (2 ** attempt)  # Exponential backoff
                log.debug(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
        
        log.error(f"Failed to fetch {url} after {config.retry_attempts + 1} attempts")
        return None

    @cached(ttl=86400, cache=Cache.MEMORY, serializer=JsonSerializer())
    async def do_know(self, norma_visitata: NormaVisitata) -> Optional[Tuple[str, str]]:
        log.info(f"Checking if knowledge exists for norma: {norma_visitata}")

        norma_str: Optional[str] = self._build_norma_string(norma_visitata)
        if norma_str is None:
            log.error("Invalid norma format")
            raise ValueError("Invalid norma format")

        search_str = norma_str.lower()
        for txt, link in self.knowledge[0].items():
            if search_str in txt.lower():
                log.info(f"Knowledge found for norma: {norma_visitata}")
                return txt, link

        log.warning(f"No knowledge found for norma: {norma_visitata}")
        return None

    @cached(ttl=86400, cache=Cache.MEMORY, serializer=JsonSerializer())
    async def look_up(self, norma_visitata: NormaVisitata) -> Optional[str]:
        log.info(f"Looking up norma: {norma_visitata}")

        norma_info = await self.do_know(norma_visitata)
        if not norma_info:
            log.info("No norma info found in knowledge base")
            return None

        link: str = norma_info[1]
        
        session = await http_client.get_session()
        log.info(f"Requesting main link: {link}")
        html_text = await self._make_request_with_retry(session, link)
        if not html_text:
            log.error(f"Failed to retrieve content for norma link: {link}")
            return None
            
        soup: BeautifulSoup = BeautifulSoup(html_text, 'html.parser')

        numero_articolo: Optional[str] = (
            norma_visitata.numero_articolo.replace('-', '')
            if norma_visitata.numero_articolo else None
        )
        
        if numero_articolo:
            article_link = await self._find_article_link(soup, BASE_URL, numero_articolo, session)
            return article_link
            
        log.info("No article number provided")
        return None

    async def _find_article_link(self, soup: BeautifulSoup, base_url: str, numero_articolo: str, 
                                 session: aiohttp.ClientSession) -> Optional[str]:
        """
        Trova il link dell'articolo con logica migliorata per ridurre le richieste.
        """
        # Compila il pattern una sola volta
        pattern = re.compile(rf'href=["\']([^"\']*art{re.escape(numero_articolo)}\.html)["\']')
        log.info(f"Searching for article {numero_articolo} in the main page content")

        # Prima ricerca: nella pagina principale
        matches = pattern.findall(str(soup))
        if matches:
            found_url = requests.compat.urljoin(base_url, matches[0])
            log.info(f"Direct match found: {found_url}")
            return found_url

        # Seconda ricerca: nelle sezioni, ma con limit e strategia migliorata
        log.info("No direct match found, searching in section links with rate limiting")
        section_titles = soup.find_all('div', class_='section-title')
        
        if not section_titles:
            log.warning("No section-title divs found")
            return None

        # Raccogliamo tutti i link delle sezioni
        section_links = []
        for section in section_titles:
            for a_tag in section.find_all('a', href=True):
                href = a_tag.get('href', '')
                if href:  # Solo link validi
                    full_link = requests.compat.urljoin(base_url, href)
                    section_links.append(full_link)

        # Limitiamo il numero di sezioni da controllare per evitare troppe richieste
        max_sections_to_check = min(len(section_links), 10)  # Massimo 10 sezioni
        section_links = section_links[:max_sections_to_check]
        
        log.info(f"Checking {len(section_links)} section links for article {numero_articolo}")

        # Processiamo le richieste in batch per controllare il carico
        batch_size = 3
        for i in range(0, len(section_links), batch_size):
            batch = section_links[i:i + batch_size]
            tasks = [self._check_section_for_article(session, link, pattern, base_url) 
                    for link in batch]
            
            try:
                # Usiamo timeout per evitare che si blocchi indefinitamente
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), 
                    timeout=30.0
                )
                
                # Controlliamo i risultati
                for result in results:
                    if isinstance(result, str) and result:  # Link trovato
                        log.info(f"Article link found in section: {result}")
                        return result
                    elif isinstance(result, Exception):
                        log.debug(f"Exception in section check: {result}")
                        
            except asyncio.TimeoutError:
                log.warning(f"Timeout while checking batch {i//batch_size + 1}")
                continue
            except Exception as e:
                log.error(f"Unexpected error in batch {i//batch_size + 1}: {e}")
                continue

            # Piccola pausa tra i batch per essere rispettosi verso il server
            if i + batch_size < len(section_links):
                await asyncio.sleep(0.5)

        log.info(f"No matching article found for {numero_articolo} after checking sections")
        return None

    async def _check_section_for_article(self, session: aiohttp.ClientSession, section_url: str, 
                                         pattern: re.Pattern, base_url: str) -> Optional[str]:
        """
        Controlla una singola sezione per l'articolo cercato.
        """
        try:
            log.debug(f"Checking section: {section_url}")
            html_content = await self._make_request_with_retry(session, section_url)
            
            if html_content:
                matches = pattern.findall(html_content)
                if matches:
                    found_url = requests.compat.urljoin(base_url, matches[0])
                    log.debug(f"Article found in section {section_url}: {found_url}")
                    return found_url
                    
        except Exception as e:
            log.debug(f"Error checking section {section_url}: {e}")
            
        return None

    async def get_info(self, norma_visitata: NormaVisitata) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
        """
        Ottiene le informazioni della norma con gestione errori migliorata.
        """
        log.info(f"Getting info for norma: {norma_visitata}")

        try:
            norma_link = await self.look_up(norma_visitata)
            if not norma_link:
                log.info("No norma link found")
                return None, {}, None

            session = await http_client.get_session()
            html_text = await self._make_request_with_retry(session, norma_link)
            
            if not html_text:
                log.error(f"Failed to retrieve content for norma link: {norma_link}")
                return None, {}, None
                
            soup = BeautifulSoup(html_text, 'html.parser')

            info: Dict[str, Any] = {}
            info['Position'] = self._extract_position(soup)
            self._extract_sections(soup, info)
            
            log.info(f"Successfully extracted info for norma: {norma_visitata}")
            return info.get('Position'), info, norma_link
            
        except Exception as e:
            log.error(f"Unexpected error in get_info for {norma_visitata}: {e}")
            return None, {}, None

    def _extract_position(self, soup: BeautifulSoup) -> Optional[str]:
        """Estrae la posizione dal breadcrumb."""
        try:
            position_tag = soup.find('div', id='breadcrumb', recursive=True)
            if position_tag:
                # Mantiene la logica originale di slicing
                position = position_tag.get_text(strip=False).replace('\n', '').replace('  ', '')[17:]
                return position if position.strip() else None
        except Exception as e:
            log.warning(f"Error extracting position: {e}")
        
        log.warning("Breadcrumb position not found")
        return None

    def _extract_sections(self, soup: BeautifulSoup, info: Dict[str, Any]) -> None:
        """Estrae le sezioni del contenuto con gestione errori migliorata."""
        try:
            corpo = soup.find('div', class_='panes-condensed panes-w-ads content-ext-guide content-mark', recursive=True)
            if not corpo:
                log.warning("Main content section not found")
                return

            # Estrazione Brocardi
            try:
                brocardi_sections = corpo.find_all('div', class_='brocardi-content')
                if brocardi_sections:
                    info['Brocardi'] = [section.get_text(strip=False) for section in brocardi_sections]
            except Exception as e:
                log.warning(f"Error extracting Brocardi sections: {e}")

            # Estrazione Ratio
            try:
                ratio_section = corpo.find('div', class_='container-ratio')
                if ratio_section:
                    ratio_text = ratio_section.find('div', class_='corpoDelTesto')
                    if ratio_text:
                        info['Ratio'] = ratio_text.get_text(strip=False)
            except Exception as e:
                log.warning(f"Error extracting Ratio section: {e}")

            # Estrazione Spiegazione
            try:
                spiegazione_header = corpo.find('h3', string=lambda text: text and "Spiegazione dell'art" in text)
                if spiegazione_header:
                    spiegazione_content = spiegazione_header.find_next_sibling('div', class_='text')
                    if spiegazione_content:
                        info['Spiegazione'] = spiegazione_content.get_text(strip=False)
            except Exception as e:
                log.warning(f"Error extracting Spiegazione section: {e}")

            # Estrazione Massime
            try:
                massime_header = corpo.find('h3', string=lambda text: text and "Massime relative all'art" in text)
                if massime_header:
                    massime_content = massime_header.find_next_sibling('div', class_='text')
                    if massime_content:
                        info['Massime'] = [massima.get_text(strip=False) for massima in massime_content]
            except Exception as e:
                log.warning(f"Error extracting Massime section: {e}")
                
        except Exception as e:
            log.error(f"Unexpected error in _extract_sections: {e}")

    def _build_norma_string(self, norma_visitata: Union[NormaVisitata, str]) -> Optional[str]:
        """Costruisce la stringa della norma per la ricerca."""
        try:
            if isinstance(norma_visitata, NormaVisitata):
                norma = norma_visitata.norma
                tipo_norm = normalize_act_type(norma.tipo_atto_str, True, 'brocardi')
                components = [tipo_norm]
                if norma.data:
                    components.append(f"{norma.data},")
                if norma.numero_atto:
                    components.append(f"n. {norma.numero_atto}")
                return " ".join(components).strip()
            elif isinstance(norma_visitata, str):
                return norma_visitata.strip()
        except Exception as e:
            log.error(f"Error building norma string: {e}")
        
        return None
