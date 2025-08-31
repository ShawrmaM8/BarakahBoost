# utils/validation.py
from __future__ import annotations
from datetime import datetime
from typing import Dict, Any

PRAYERS = ["Fajr", "Dhuhr", "Asr", "Maghrib", "Isha"]


def parse_date(date_str: str) -> str:
    # Normalize date to YYYY-MM-DD
    return datetime.fromisoformat(date_str).date().isoformat()


def clean_outcomes(o: dict[str, Any]) -> dict[str, int]:
    out = {}
    for k in ["clarity", "focus", "calm", "productivity"]:
        v = o.get(k)
        if v is None:
            continue
        try:
            v = int(v)
        except (TypeError, ValueError):
            continue
        out[k] = max(1, min(5, v))
    return out



def validate_prayers(p: Dict[str, Any]) -> Dict[str, int]:
    out = {}
    for name in PRAYERS:
        out[name] = 1 if bool(p.get(name, False)) else 0
    return out