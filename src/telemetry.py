"""Observability: structured JSON logging plus per-query traces and user
feedback appended as JSON lines under logs/.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field

import config


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if any(isinstance(h.formatter, _JsonFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)


@dataclass
class QueryTrace:
    query: str
    strategy: str
    reranked: bool
    refused: bool
    route: str = ""  # router decision: smalltalk|comparison|lexical|semantic|hybrid
    # Guardrails: which input guard fired ("" if none — injection|advice|length),
    # and any [n] citations the answer made that point at no real source.
    guardrail: str = ""
    dangling_citations: list[int] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)  # [{chunk_id, score}]
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    est_cost_usd: float = 0.0
    # Reproducibility provenance: which model snapshot + backend build answered.
    model: str = ""
    system_fingerprint: str | None = None
    ts: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = config.MODEL_PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def _append_jsonl(path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def log_query(trace: QueryTrace) -> None:
    _append_jsonl(config.QUERY_LOG, asdict(trace))


def log_feedback(record: dict) -> None:
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    _append_jsonl(config.FEEDBACK_LOG, record)
