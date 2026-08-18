"""Webhook authentication.

TradingView can't send custom headers, so the shared secret travels inside
the JSON body instead and is checked here with a constant-time comparison.
"""
import hmac

from app.config import settings

PLACEHOLDER_SECRET = "changeme-generate-a-long-random-string"


def verify_webhook_secret(provided: str) -> bool:
    if not settings.WEBHOOK_SECRET or settings.WEBHOOK_SECRET == PLACEHOLDER_SECRET:
        # Refuse everything until the deployer sets a real secret — an
        # unauthenticated webhook that can trigger trades is not safe to run.
        return False
    return hmac.compare_digest(provided or "", settings.WEBHOOK_SECRET)
