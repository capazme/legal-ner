import asyncio
import json
from quart import Blueprint, request, jsonify, Response
import structlog

from visualex_api.services.eurlex_scraper import EurlexScraper
from visualex_api.services.normattiva_scraper import NormattivaScraper
from visualex_api.tools.norma import Norma, NormaVisitata
from visualex_api.tools.text_op import format_date_to_extended, parse_article_input
from visualex_api.tools.urngenerator import complete_date_or_parse

log = structlog.get_logger()

norma_bp = Blueprint('norma', __name__)

normattiva_scraper = NormattivaScraper()
eurlex_scraper = EurlexScraper()

def get_scraper_for_norma(normavisitata):
    act_type_normalized = normavisitata.norma.tipo_atto.lower()
    log.debug("Determining scraper for norma", act_type=act_type_normalized)
    if act_type_normalized in ['tue', 'tfue', 'cdfue', 'regolamento ue', 'direttiva ue']:
        return eurlex_scraper
    else:
        return normattiva_scraper

async def create_norma_visitata_from_data(data):
    log.info("Creating NormaVisitata from data", data=data)
    allowed_types = ['legge', 'decreto legge', 'decreto legislativo', 'd.p.r.', 'regio decreto']
    act_type = data.get('act_type')
    if act_type in allowed_types:
        log.info("Act type is allowed", act_type=act_type)
        data_completa = complete_date_or_parse(
            date=data.get('date'),
            act_type=act_type,
            act_number=data.get('act_number')
        )
        log.info("Completed date parsed", data_completa=data_completa)
        data_completa_estesa = format_date_to_extended(data_completa)
        log.info("Extended date formatted", data_completa_estesa=data_completa_estesa)
    else:
        log.info("Act type is not in allowed types", act_type=act_type)
        data_completa_estesa = data.get('date')
        log.info("Using provided date", data_completa_estesa=data_completa_estesa)

    norma = Norma(
        tipo_atto=act_type,
        data=data_completa_estesa if data_completa_estesa else None,
        numero_atto=data.get('act_number')
    )
    log.info("Norma instance created", norma=norma)

    articles = await parse_article_input(str(data.get('article')), norma.url)
    log.info("Articles parsed", articles=articles)

    norma_visitata_list = []
    for article in articles:
        cleaned_article = article.strip().replace(' ', '-') if ' ' in article.strip() else article.strip()
        log.info("Processing article", article=cleaned_article)
        norma_visitata_list.append(NormaVisitata(
            norma=norma,
            numero_articolo=cleaned_article,
            versione=data.get('version'),
            data_versione=data.get('version_date'),
            allegato=data.get('annex')
        ))
        log.info("NormaVisitata instance created", norma_visitata=norma_visitata_list[-1])

    log.info("Created NormaVisitata instances", norma_visitata_list=[nv.to_dict() for nv in norma_visitata_list])
    return norma_visitata_list

@norma_bp.route('/fetch_norma_data', methods=['POST'])
async def fetch_norma_data():
    try:
        data = await request.get_json()
        log.info("Received data for fetch_norma_data", data=data)

        normavisitate = await create_norma_visitata_from_data(data)
        response = {'norma_data': [nv.to_dict() for nv in normavisitate]}
        log.debug("Norma data response", response=response)
        return jsonify(response)
    except Exception as e:
        log.error("Error in fetch_norma_data", error=str(e))
        return jsonify({'error': str(e)}), 500

@norma_bp.route('/fetch_article_text', methods=['POST'])
async def fetch_article_text():
    try:
        data = await request.get_json()
        log.info("Received data for fetch_article_text", data=data)

        normavisitate = await create_norma_visitata_from_data(data)
        log.info("NormaVisitata instances created", normavisitate=[nv.to_dict() for nv in normavisitate])

        async def fetch_text(nv):
            scraper = get_scraper_for_norma(nv)
            if scraper is None:
                log.warning("Unsupported act type for scraper", norma_data=nv.to_dict())
                return {'error': 'Unsupported act type', 'norma_data': nv.to_dict()}

            try:
                article_text, url = await scraper.get_document(nv)
                log.info("Document fetched successfully", article_text=article_text, url=url)
                return {
                    'article_text': article_text,
                    'norma_data': nv.to_dict(),
                    'url': url
                }
            except Exception as exc:
                log.error("Error fetching article text", error=str(exc))
                return {'error': str(exc), 'norma_data': nv.to_dict()}

        results = await asyncio.gather(*(fetch_text(nv) for nv in normavisitate), return_exceptions=True)
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                log.error("Exception during fetching article text", exception=str(result))
                processed_results.append({'error': str(result)})
            else:
                processed_results.append(result)
                log.info("Fetched article result", result=result)
        return jsonify(processed_results)
    except Exception as e:
        log.error("Error in fetch_article_text", error=str(e))
        return jsonify({'error': str(e)}), 500

@norma_bp.route('/stream_article_text', methods=['POST'])
async def stream_article_text():
    data = await request.get_json()
    log.info("Received data for stream_article_text", data=data)
    normavisitate = await create_norma_visitata_from_data(data)
    log.info("NormaVisitata instances created", normavisitate=[nv.to_dict() for nv in normavisitate])
    
    async def result_generator():
        for nv in normavisitate:
            scraper = get_scraper_for_norma(nv)
            if scraper is None:
                result = {'error': 'Unsupported act type', 'norma_data': nv.to_dict()}
                yield json.dumps(result) + "\n"
                continue

            try:
                article_text, url = await scraper.get_document(nv)
                result = {
                    'article_text': article_text,
                    'norma_data': nv.to_dict(),
                    'url': url
                }
            except Exception as exc:
                result = {'error': str(exc), 'norma_data': nv.to_dict()}
            yield json.dumps(result) + "\n"
            await asyncio.sleep(0.05)
    
    return Response(result_generator(), mimetype="application/json")
