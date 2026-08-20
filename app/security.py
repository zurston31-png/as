"""Webhook authentication.

TradingView can't send custom headers, so the shared secret travels inside
the JSON body instead and is checked here with a constant-time comparison.
"""
import hmac

from app.config import settings

PLACEHOLDER_SECRET = "changeme-generate-a-long-random-string"


def secret_is_configured() -> bool:
    """Whether this deployment has a real secret at all.

    Worth asking separately, because the two ways to fail authentication
    need opposite fixes and are indistinguishable from the caller's side.
    The Pine script ships the SAME placeholder as its default input, so a
    deployer who has set neither end has two values that match exactly and
    still get refused - and a log saying only "invalid secret" sends them
    to check TradingView, which is the half that is already correct.
    """
    return bool(settings.WEBHOOK_SECRET) and settings.WEBHOOK_SECRET != PLACEHOLDER_SECRET


def verify_webhook_secret(provided: str) -> bool:
    if not secret_is_configured():
        # Refuse everything until the deployer sets a real secret — an
        # unauthenticated webhook that can trigger trades is not safe to run.
        return False
    return hmac.compare_digest(provided or "", settings.WEBHOOK_SECRET)
