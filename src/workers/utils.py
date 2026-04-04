from __future__ import annotations

import hashlib
from datetime import datetime

from dateutil import parser as date_parser


def parse_published_at(value: str | None) -> datetime | None:
    if not value:
        return None
    return date_parser.isoparse(value)


def build_external_id(payload: dict) -> str:
    if payload.get("external_id"):
        return str(payload["external_id"])

    title = payload.get("title") or ""
    source = payload.get("source") or ""
    published_at = payload.get("published_at") or ""
    text = payload.get("text") or ""

    digest_source = f"{title}|{source}|{published_at}|{text}"
    return hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
