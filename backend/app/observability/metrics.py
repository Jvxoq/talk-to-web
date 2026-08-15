"""One structured log line per reply, not a Prometheus counter.

An in-process counter is wrong the moment there is more than one worker: each
process would report its own slice of traffic, and nothing would ever add them
back together. It is also wrong for this deployment specifically, because
nothing scrapes a `/metrics` endpoint here - there is no Prometheus pointed at
this single container to pull one from. A log line survives both problems: it
already goes somewhere that aggregates across processes, and it costs nothing
to add now.

This is not the final form. The day a real scraper exists, the call site below
becomes a counter increment (or both, during the migration) - this module is
the placeholder for that, not a decision that logging is enough forever.
"""

from loguru import logger


def emit_reply_metrics(**fields: object) -> None:
    logger.bind(event="chat.reply", **fields).info("chat.reply")
