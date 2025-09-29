import asyncio
import os
from quart import Blueprint, request, jsonify, send_file
import structlog

from visualex_api.services.brocardi_scraper import BrocardiScraper
from visualex_api.services.pdfextractor import extract_pdf
from visualex_api.tools.sys_op import WebDriverManager
from visualex_api.tools.urngenerator import urn_to_filename
from visualex_api.tools.treextractor import get_tree
from visualex_api.routes.norma import create_norma_visitata_from_data, get_scraper_for_norma

log = structlog.get_logger()

extra_bp = Blueprint('extra', __name__)

brocardi_scraper = BrocardiScraper()
driver_manager = WebDriverManager()

@extra_bp.route('/fetch_all_data', methods=['POST'])
async def fetch_all_data():
    try:
        data = await request.get_json()
        log.info("Received data for fetch_all_data", data=data)

        normavisitate = await create_norma_visitata_from_data(data)

        async def fetch_data(nv):
            scraper = get_scraper_for_norma(nv)
            if scraper is None:
                log.warning("Unsupported act type for scraper", norma_data=nv.to_dict())
                return {'error': 'Unsupported act type', 'norma_data': nv.to_dict()}

            try:
                article_text, url = await scraper.get_document(nv)
                brocardi_info = None
                if scraper.__class__.__name__ == 'NormattivaScraper':
                    try:
                        b_info = await brocardi_scraper.get_info(nv)
                        brocardi_info = {
                            'position': b_info[0] if b_info[0] else None,
                            'link': b_info[2],
                            'Brocardi': b_info[1].get('Brocardi') if b_info[1] and 'Brocardi' in b_info[1] else None,
                            'Ratio': b_info[1].get('Ratio') if b_info[1] and 'Ratio' in b_info[1] else None,
                            'Spiegazione': b_info[1].get('Spiegazione') if b_info[1] and 'Spiegazione' in b_info[1] else None,
                            'Massime': b_info[1].get('Massime') if b_info[1] and 'Massime' in b_info[1] else None
                        }
                    except Exception as exc:
                        log.error("Error fetching Brocardi info", error=str(exc))
                        brocardi_info = {'error': str(exc)}
                return {
                    'article_text': article_text,
                    'url': url,
                    'norma_data': nv.to_dict(),
                    'brocardi_info': brocardi_info
                }
            except Exception as exc:
                log.error("Error fetching all data", error=str(exc))
                return {'error': str(exc), 'norma_data': nv.to_dict()}

        results = await asyncio.gather(*(fetch_data(nv) for nv in normavisitate), return_exceptions=True)
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({'error': str(result)})
                log.error("Exception during fetching all data", exception=str(result))
            else:
                processed_results.append(result)
        return jsonify(processed_results)
    except Exception as e:
        log.error("Error in fetch_all_data", error=str(e))
        return jsonify({'error': str(e)}), 500

@extra_bp.route('/fetch_tree', methods=['POST'])
async def fetch_tree():
    try:
        data = await request.get_json()
        log.info("Received data for fetch_tree", data=data)

        urn = data.get('urn')
        if not urn:
            log.error("Missing 'urn' in request data")
            return jsonify({'error': "Missing 'urn' in request data"}), 400

        link = data.get('link', False)
        details = data.get('details', False)
        if not isinstance(link, bool):
            log.error("'link' must be a boolean")
            return jsonify({'error': "'link' must be a boolean"}), 400
        if not isinstance(details, bool):
            log.error("'details' must be a boolean")
            return jsonify({'error': "'details' must be a boolean"}), 400

        log.debug("Flags received", link=link, details=details)
        articles, count = await get_tree(urn, link=link, details=details)
        if isinstance(articles, str):
            log.error("Error fetching tree", error=articles)
            return jsonify({'error': articles}), 500

        response = {'articles': articles, 'count': count}
        log.info("Tree fetched successfully", response=response)
        return jsonify(response)
    except Exception as e:
        log.error("Error in fetch_tree", error=str(e), exc_info=True)
        return jsonify({'error': str(e)}), 500

@extra_bp.route('/history', methods=['GET'])
async def get_history():
    try:
        return jsonify({'history': list(history)})
    except Exception as e:
        log.error("Error in get_history", error=str(e))
        return jsonify({'error': str(e)}), 500

@extra_bp.route('/export_pdf', methods=['POST'])
async def export_pdf():
    try:
        data = await request.get_json()
        urn = data.get('urn')
        if not urn:
            return jsonify({'error': 'URN mancante'}), 400

        log.info("Received data for export_pdf", data=data)
        pdf_path = urn_to_filename(urn)

        file_exists = await asyncio.to_thread(os.path.exists, pdf_path)
        if file_exists:
            file_size = await asyncio.to_thread(os.path.getsize, pdf_path)
            if file_size > 0:
                log.info(f"File PDF già presente e valido: {pdf_path}. Serve file cache.")
                return await send_file(
                    pdf_path,
                    mimetype='application/pdf',
                    as_attachment=True,
                    attachment_filename=os.path.basename(pdf_path)
                )
            else:
                log.info(f"File PDF presente ma vuoto: {pdf_path}. Rimuovo e rigenero.")
                await asyncio.to_thread(os.remove, pdf_path)

        driver = await asyncio.to_thread(driver_manager.setup_driver)
        try:
            extracted_pdf_path = await asyncio.to_thread(extract_pdf, driver, urn)
            log.info(f"PDF estratto: {extracted_pdf_path}")
        finally:
            await asyncio.to_thread(driver_manager.close_drivers)

        exists_extracted = await asyncio.to_thread(os.path.exists, extracted_pdf_path)
        size_extracted = await asyncio.to_thread(os.path.getsize, extracted_pdf_path) if exists_extracted else 0
        if not exists_extracted or size_extracted == 0:
            raise Exception("Il PDF estratto risulta vuoto o non esistente.")

        if extracted_pdf_path != pdf_path:
            def copy_file(src, dst):
                with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                    fdst.write(fsrc.read())
            await asyncio.to_thread(copy_file, extracted_pdf_path, pdf_path)
            log.info(f"PDF copiato in cache: {pdf_path}")
        else:
            log.info("PDF estratto usato come cache.")

        return await send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            attachment_filename=os.path.basename(pdf_path)
        )
    except Exception as e:
        log.error("Error in export_pdf", error=str(e))
        return jsonify({'error': str(e)}), 500
