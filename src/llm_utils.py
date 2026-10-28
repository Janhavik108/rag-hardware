"""
Shared LLM-call helper: wraps client.chat.completions.create with retry +
exponential backoff on rate limit errors.

Groq's free tier has a low tokens-per-minute cap -- as low as 8000 TPM on
some accounts -- and even a single eval question (draft call + possible tool
follow-up + verification call, each carrying several retrieved excerpts) can
burn a few thousand tokens. A 20-30 question eval run hits this regularly.
Without a retry here, eval.py crashes on whichever question first exceeds
the limit and loses all progress on the questions already completed.

Only rate limits are retried. A real error (bad model name, malformed
request, auth failure) fails the same way on every attempt, so retrying it
just wastes time -- it's re-raised immediately.
"""
import time

try:
    from openai import RateLimitError
except ImportError:
    RateLimitError = ()  # openai not installed -- `except ()` matches nothing,
    # so callers with an injected fake client (tests) are unaffected either way.


def create_completion(client, max_retries: int = 6, base_delay: float = 8.0, **kwargs):
    """client.chat.completions.create(**kwargs), retried with exponential
    backoff (base_delay, 2x, 4x, ...) if the provider returns a rate-limit
    error. Prints a heads-up on each retry so a long eval run doesn't look
    silently stuck."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            last_error = e
            delay = base_delay * (2 ** attempt)
            print(f"[llm] rate limited, waiting {delay:.0f}s before retry "
                  f"{attempt + 1}/{max_retries}...")
            time.sleep(delay)
    raise last_error
