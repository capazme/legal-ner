from quart import Blueprint, render_template, send_from_directory

docs_bp = Blueprint('docs', __name__, static_folder='../static', template_folder='../templates')

@docs_bp.route('/docs')
async def get_docs():
    return await render_template('swagger_ui.html')

@docs_bp.route('/openapi.json')
async def openapi_json():
    return await send_from_directory(docs_bp.static_folder, 'swagger.yaml')
