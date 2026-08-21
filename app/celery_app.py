"""
Celery application factory.

Broker and result backend both point to Redis.
"""

from celery import Celery

from app.config import settings

celery_app = Celery(
    "rag_engine",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    # ── Serialization ──
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # ── Visibility ──
    task_track_started=True,         # expose STARTED state
    result_expires=3600,             # results live for 1 hour

    # ── Worker tuning ──
    worker_prefetch_multiplier=1,    # fair scheduling for long tasks
    task_acks_late=True,             # ack only after completion

    # ── Task discovery ──
    include=["app.tasks"],
)
