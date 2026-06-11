"""Shared retry/backoff decorators (tenacity) for every external call site:
SEC EDGAR (requests), OpenAI (embeddings), Anthropic (generation + judge), and
Pinecone. SDK clients are constructed with max_retries=0 so tenacity owns the
retry policy and we don't double-backoff.
"""
from __future__ import annotations

import logging

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger("reliability")

HTTP_TIMEOUT = 30
OPENAI_TIMEOUT = 60
ANTHROPIC_TIMEOUT = 120


def _openai_errors():
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    return (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


def _anthropic_errors():
    from anthropic import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        OverloadedError,
        RateLimitError,
    )

    return (
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
        OverloadedError,
    )


def _pinecone_errors():
    from pinecone.exceptions import PineconeApiException

    return (PineconeApiException, requests.RequestException, ConnectionError, TimeoutError)


http_retry = retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
    before_sleep=before_sleep_log(log, logging.WARNING),
)


def openai_retry(fn):
    # 8 attempts (waits 1,2,4,8,16,32,60,60s ≈ 3 min of budget) so a sustained
    # token-per-minute rate-limit burst — which can last a full 60s window — is
    # ridden out instead of exhausting the retries and crashing a long eval run.
    # The org's 30K TPM ceiling makes heavy judge passes the main trigger.
    return retry(
        retry=retry_if_exception_type(_openai_errors()),
        wait=wait_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(8),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.WARNING),
    )(fn)


def anthropic_retry(fn):
    # Same budget as the OpenAI path: 8 attempts (~3 min) so a sustained
    # token-per-minute rate-limit burst is ridden out instead of crashing a long
    # eval/generation run. Generation (Opus) and the judge (Sonnet) run on
    # separate per-model rate-limit buckets, so they no longer compete.
    return retry(
        retry=retry_if_exception_type(_anthropic_errors()),
        wait=wait_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(8),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.WARNING),
    )(fn)


def pinecone_retry(fn):
    return retry(
        retry=retry_if_exception_type(_pinecone_errors()),
        wait=wait_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.WARNING),
    )(fn)


def make_openai_client():
    import config

    if not config.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    from openai import OpenAI

    return OpenAI(api_key=config.OPENAI_API_KEY, timeout=OPENAI_TIMEOUT, max_retries=0)


def make_anthropic_client():
    import config

    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env "
            "(see .env.example) — generation and the LLM judge run on Claude."
        )
    from anthropic import Anthropic

    return Anthropic(
        api_key=config.ANTHROPIC_API_KEY, timeout=ANTHROPIC_TIMEOUT, max_retries=0
    )
