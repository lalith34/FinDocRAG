"""Shared retry/backoff decorators (tenacity) for every external call site:
SEC EDGAR (requests), OpenAI, and Pinecone. OpenAI clients are constructed with
max_retries=0 so tenacity owns the retry policy and we don't double-backoff.
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


def _openai_errors():
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    return (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


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
    return retry(
        retry=retry_if_exception_type(_openai_errors()),
        wait=wait_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(5),
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
