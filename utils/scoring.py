# utils/scoring.py
from __future__ import annotations
import math
from datetime import datetime, time
from typing import Dict, Any, List
from data.reference.surah_meta import NAME_TO_AYAH

# --- Helper scalers (all bounded 0..100 for component scores) ---

def minmax(value, min_v, max_v):
    if max_v <= min_v:
        return 0.0
    v = (value - min_v) / (max_v - min_v)
    return float(max(0.0, min(1.0, v))) * 100.0

def inverse_minmax(value, min_v, max_v):
    # High value -> low score
    return (100.0 - minmax(value, min_v, max_v))

# --- Component scoring functions ---

def score_prayer(prayers: Dict[str, int]) -> float:
    # Each on-time prayer contributes equally
    if not prayers:
        return 0.0
    total = sum(1 for _ in prayers)
    done = sum(int(v) for v in prayers.values())
    return (done / total) * 100.0

def score_quran(recitations: List[Dict[str, Any]], base_points_per_ayah: float,
                max_daily_points: int) -> float:
    if not recitations:
        return 0.0
    pts = 0.0
    for r in recitations:
        sname = r.get("surah")
        ayat = r.get("ayahs", 0)  # explicit ayahs read today (optional)
        if not ayat:
            # fallback to whole surah if specified and ayahs not provided
            ayat = NAME_TO_AYAH.get(sname, 0)
        pts += ayat * base_points_per_ayah
    return min(100.0, (pts / max_daily_points) * 100.0)

def score_dhikr(total_reps: int, points_per_repetition: float, max_daily_points: int) -> float:
    pts = total_reps * points_per_repetition
    return min(100.0, (pts / max_daily_points) * 100.0)

def score_sadaqah(amount: float, log_scale: bool, log_base: float,
                  max_daily_points: int) -> float:
    amount = max(0.0, float(amount or 0.0))
    if amount <= 0:
        return 0.0
    if log_scale:
        val = math.log(amount + 1, log_base)
    else:
        val = amount
    return min(
        100.0,
        (val / (math.log(1000 + 1, log_base) if log_scale else 1000.0)) * 100.0
    )

def _to_time(tstr: str) -> time:
    return datetime.strptime(tstr, "%H:%M").time()

def score_sleep(hours: float, bedtime: str, ideal_min: float, ideal_max: float,
                bedtime_bonus_before: str, max_daily_points: int) -> float:
    core = 0.0
    if hours is not None:
        # Triangular score peaking in [ideal_min, ideal_max]
        if hours < ideal_min:
            core = minmax(hours, ideal_min - 3, ideal_min)
        elif hours > ideal_max:
            core = inverse_minmax(hours, ideal_max, ideal_max + 3)
        else:
            core = 100.0
    bonus = 0.0
    try:
        if bedtime and _to_time(bedtime) <= _to_time(bedtime_bonus_before):
            bonus = 10.0
    except Exception:
        pass
    return min(100.0, max(0.0, core) + bonus)

def score_screen_time(app_minutes: Dict[str, float], cfg: Dict[str, Any]) -> float:
    if not app_minutes:
        return 50.0  # neutral when unknown
    prod = sum(m for a, m in app_minutes.items() if a in cfg["productive_apps"])  # minutes
    dist = sum(m for a, m in app_minutes.items() if a in cfg["distracting_apps"])  # minutes
    total = prod + dist
    total = max(total, 1.0)
    # Aim: maximize productive ratio and minimize distracting minutes
    ratio = prod / total
    # Cap distracting minutes against a max window
    dist_penalty = 1.0 - min(1.0, dist / max(1.0, float(cfg["max_daily_minutes"])))
    combined = 0.6 * ratio + 0.4 * dist_penalty
    return float(max(0.0, min(1.0, combined))) * 100.0

def score_other(good_count: int, bad_count: int, good_points: int, bad_points: int) -> float:
    pos = good_count * good_points
    neg = bad_count * bad_points
    raw = pos - neg
    # Map [-100, 100] -> [0, 100]
    return float(max(0.0, min(100.0, (raw + 100) / 200 * 100)))

def weighted_baraka_score(components: Dict[str, float], weights: Dict[str, float]) -> float:
    # Ensure weights sum to 1
    sw = sum(weights.values()) or 1.0
    return sum(components.get(k, 0.0) * (w / sw) for k, w in weights.items())
