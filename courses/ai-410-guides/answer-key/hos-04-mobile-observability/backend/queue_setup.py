"""
Queue connection — complete as-is.
"""

import os

from dotenv import load_dotenv
from redis import Redis
from rq import Queue

load_dotenv()

redis_conn = Redis.from_url(os.environ["REDIS_URL"])
queue = Queue("ai410-ingestion", connection=redis_conn)
