"""One structured log line per reply, not a Prometheus counter.

Nothing scrapes `/metrics` here, and logs already aggregate across processes.
Becomes a counter once a scraper exists.
"""

from loguru import logger


def emit_reply_metrics(**fields: object) -> None:
    logger.bind(event="chat.reply", **fields).info("chat.reply")
