import os

MAX_CACHE_SIZE = int(os.getenv('MAX_CACHE_SIZE', 10000))
HISTORY_LIMIT = int(os.getenv('HISTORY_LIMIT', 50))
RATE_LIMIT = int(os.getenv('RATE_LIMIT', 1000))
RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', 600))