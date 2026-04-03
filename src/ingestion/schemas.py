from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class IncomingNewsMessage:
    external_id: str
    title: Optional[str]
    source: Optional[str]
    published_at: Optional[str]
    text: str
    url: Optional[str] = None
    payload: Optional[dict[str, Any]] = None
