import asyncio
import os
import json
from collections import defaultdict, deque
from time import time

from quart import Quart, request, jsonify, g
from quart_cors import cors
import structlog

from visualex_api.tools.logger_config import setup_logging
from visualex_api.tools.http_client import http_client
from visualex_api.tools.sys_op import web_driver_manager
from visualex_api.tools.config import HISTORY_LIMIT, RATE_LIMIT, RATE_LIMIT_WINDOW
from visualex_api.routes.norma import norma_bp
from visualex_api.routes.brocardi import brocardi_bp
from visualex_api.routes.extra import extra_bp
from visualex_api.routes.docs import docs_bp

# Setup logging
setup_logging()
log = structlog.get_logger()

# Funzione per il conteggio dei token (numero di parole) in modo ricorsivo
def count_tokens(data):
    if isinstance(data, str):
        return len(data.split())
    elif isinstance(data, dict):
        return sum(count_tokens(v) for v in data.values())
    elif isinstance(data, list):
        return sum(count_tokens(item) for item in data)
    else:
        return 0

# Storage per il rate limiting e la history
request_counts = defaultdict(lambda: {'count': 0, 'time': time()})
history = deque(maxlen=HISTORY_LIMIT)

def create_app():
    app = Quart(__name__)
    app = cors(app, allow_origin="*")

    # Middleware per registrare il tempo di inizio della richiesta
    @app.before_request
    def record_start_time():
        g.start_time = time()

    # Middleware per il rate limiting
    @app.before_request
    def rate_limit_middleware():
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        current_time = time()
        log.debug("Rate limit check", client_ip=client_ip, current_time=current_time)

        request_info = request_counts[client_ip]
        if current_time - request_info['time'] < RATE_LIMIT_WINDOW:
            if request_info['count'] >= RATE_LIMIT:
                log.warning("Rate limit exceeded", client_ip=client_ip)
                return jsonify({'error': 'Rate limit exceeded. Try again later.'}), 429
            else:
                request_info['count'] += 1
        else:
            request_counts[client_ip] = {'count': 1, 'time': current_time}

    # Middleware per loggare statistiche (tempo e token) dopo ogni richiesta
    @app.after_request
    def log_query_stats(response):
        try:
            end_time = time()
            start_time = getattr(g, "start_time", end_time)
            duration = end_time - start_time

            tokens = None
            if response.content_type and "application/json" in response.content_type:
                # Estrae il testo della risposta e lo decodifica in JSON
                text = response.get_data(as_text=True)
                try:
                    data = json.loads(text)
                    tokens = count_tokens(data)
                except Exception:
                    tokens = "N/A"
            log.info("Query statistics", path=request.path, method=request.method, duration=duration, tokens=tokens)
        except Exception as e:
            log.error("Error logging query statistics", error=str(e))
        return response

    @app.after_serving
    async def shutdown():
        await http_client.close_session()
        web_driver_manager.close_drivers()

    # Register blueprints
    app.register_blueprint(norma_bp)
    app.register_blueprint(brocardi_bp)
    app.register_blueprint(extra_bp)
    app.register_blueprint(docs_bp)

    return app

def main():
    app = create_app()
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    app.run(host=host, port=port)

if __name__ == '__main__':
    main()