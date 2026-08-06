"""Shared outbound-HTTP helper.

Python 3.13 turned on VERIFY_X509_STRICT by default. Antivirus/corporate
TLS-inspection middleboxes commonly present root CAs whose Basic Constraints
are not marked critical, which the strict check rejects with
"certificate verify failed: Basic Constraints of CA cert not marked critical"
— on machines with such interception every HTTPS download breaks.

`urlopen_verified` keeps full certificate *verification* enabled (hostname +
chain checks stay on) and relaxes only the new strictness flag, matching how
pip/requests behave. Checksums remain the real integrity gate for model
downloads (see registry.py).
"""
from __future__ import annotations

import ssl
import urllib.request
from typing import Any


def _context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


def urlopen_verified(url_or_req: Any, timeout: float) -> Any:
    """Drop-in for urllib.request.urlopen with the relaxed-strict context."""
    return urllib.request.urlopen(url_or_req, timeout=timeout, context=_context())
