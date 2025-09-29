import asyncio
from quart import Blueprint, request, jsonify
import structlog

from visualex_api.services.brocardi_scraper import BrocardiScraper
from visualex_api.routes.norma import create_norma_visitata_from_data

log = structlog.get_logger()

brocardi_bp = Blueprint('brocardi', __name__)

brocardi_scraper = BrocardiScraper()

@brocardi_bp.route('/fetch_brocardi_info', methods=['POST'])
async def fetch_brocardi_info():
    try:
        data = await request.get_json()
        log.info("Received data for fetch_brocardi_info", data=data)

        normavisitate = await create_norma_visitata_from_data(data)

        async def fetch_info(nv):
            act_type_normalized = nv.norma.tipo_atto.lower()
            if act_type_normalized in ['tue', 'tfue', 'cdfue', 'regolamento ue', 'direttiva ue']:
                return {'norma_data': nv.to_dict(), 'brocardi_info': None}

            try:
                brocardi_info = await brocardi_scraper.get_info(nv)
                return {
                    'norma_data': nv.to_dict(),
                    'brocardi_info': {
                        'position': brocardi_info[0] if brocardi_info[0] else None,
                        'link': brocardi_info[2],
                        'Brocardi': brocardi_info[1].get('Brocardi') if brocardi_info[1] and 'Brocardi' in brocardi_info[1] else None,
                        'Ratio': brocardi_info[1].get('Ratio') if brocardi_info[1] and 'Ratio' in brocardi_info[1] else None,
                        'Spiegazione': brocardi_info[1].get('Spiegazione') if brocardi_info[1] and 'Spiegazione' in brocardi_info[1] else None,
                        'Massime': brocardi_info[1].get('Massime') if brocardi_info[1] and 'Massime' in brocardi_info[1] else None
                    }
                }
            except Exception as exc:
                log.error("Error fetching Brocardi info", error=str(exc))
                return {'error': str(exc), 'norma_data': nv.to_dict()}

        results = await asyncio.gather(*(fetch_info(nv) for nv in normavisitate), return_exceptions=True)
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({'error': str(result)})
            else:
                processed_results.append(result)
        return jsonify(processed_results)
    except Exception as e:
        log.error("Error in fetch_brocardi_info", error=str(e))
        return jsonify({'error': str(e)}), 500
