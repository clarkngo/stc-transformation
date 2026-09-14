"""
Queue connection — complete as-is. This is the "boilerplate" the
Week 7 guide refers to; you shouldn't need to touch this file.
"""

import os

from dotenv import load_dotenv
from redis import Redis
from rq import Queue

load_dotenv()

redis_conn = Redis.from_url(os.environ["REDIS_URL"])
queue = Queue("ai410-ingestion", connection=redis_conn)
