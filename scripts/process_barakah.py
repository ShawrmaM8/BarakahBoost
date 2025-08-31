# scripts/process_data.py
from __future__ import annotations
import pandas as pd
from utils.io_utils import ROOT, ensure_dirs, daily_logs_df, save_df_csv, read_config


def parse_screen_time_payload(payloads):
    """
    Accepts list of app->minutes dicts or raw dumps; merges into per-day app minutes.
    """
    per_day = {}
    for p in payloads:
        day = p.get("date")  # YYYY-MM-DD
        apps = p.get("apps", {})  # {app_name: minutes}
        if not day:
            continue
        per_day.setdefault(day, {})
        for a, m in apps.items():
            per_day[day][a] = per_day[day].get(a, 0.0) + float(m)
    return per_day


def build_features() -> pd.DataFrame:
    ensure_dirs()
    cfg = read_config()
    logs = daily_logs_df()
    if logs.empty:
        return pd.DataFrame()

    # Normalize date
    logs["date"] = pd.to_datetime(logs["date"]).dt.date.astype(str)

    # Expand prayer booleans into numeric
    prayer_cols = ["Fajr", "Dhuhr", "Asr", "Maghrib", "Isha"]
    for c in prayer_cols:
        if c not in logs.columns:
            logs[c] = 0
        logs[c] = logs[c].fillna(0).astype(int)

    # Quran recitations: list of {surah, ayahs}
    logs["quran_recs"] = logs.get("quran_recs", pd.Series([[]] * len(logs)))

    # Dhikr counts
    logs["dhikr_reps"] = logs.get("dhikr_reps", 0).fillna(0).astype(int)

    # Sadaqah amount
    logs["sadaqah_amount"] = logs.get("sadaqah_amount", 0.0).fillna(0.0).astype(float)

    # Sleep
    logs["sleep_hours"] = logs.get("sleep_hours", 0.0).fillna(0.0).astype(float)
    logs["bedtime"] = logs.get("bedtime", "").fillna("")

    # Other
    logs["other_good"] = logs.get("other_good", 0).fillna(0).astype(int)
    logs["other_bad"] = logs.get("other_bad", 0).fillna(0).astype(int)

    # Outcomes
    for k in ["clarity", "focus", "calm", "productivity"]:
        if k not in logs:
            logs[k] = None

    # Screen time payload is already summarized in app;
    # the app writes per-day app minutes into each entry
    logs["app_minutes"] = logs.get("app_minutes", [{} for _ in range(len(logs))])

    # --- Flatten to features per day ---
    feat = pd.DataFrame()
    feat["date"] = logs["date"]

    feat["prayer_on_time"] = logs[["Fajr", "Dhuhr", "Asr", "Maghrib", "Isha"]].mean(axis=1)

    feat["quran_items"] = logs["quran_recs"].apply(
        lambda xs: sum(int(x.get("ayahs", 0)) for x in xs if isinstance(xs, list))
    )

    feat["dhikr_reps"] = logs["dhikr_reps"]
    feat["sadaqah_amount"] = logs["sadaqah_amount"]
    feat["sleep_hours"] = logs["sleep_hours"]
    feat["bedtime"] = logs["bedtime"]
    feat["other_good"] = logs["other_good"]
    feat["other_bad"] = logs["other_bad"]

    # Screen time split
    def split_minutes(m):
        prod = 0.0
        dist = 0.0
        prod_list = set(cfg.get("screen_time", {}).get("productive_apps", []))
        dist_list = set(cfg.get("screen_time", {}).get("distracting_apps", []))
        for a, mins in (m or {}).items():
            if a in prod_list:
                prod += float(mins)
            if a in dist_list:
                dist += float(mins)
        return pd.Series({"prod_minutes": prod, "dist_minutes": dist})

    split = logs["app_minutes"].apply(split_minutes)
    feat = pd.concat([feat, split], axis=1)

    # Outcomes
    feat["clarity"] = logs["clarity"]
    feat["focus"] = logs["focus"]
    feat["calm"] = logs["calm"]
    feat["productivity"] = logs["productivity"]

    # Drop duplicates by date keeping last
    feat = (
        feat.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    save_df_csv(feat, "data/processed/daily_features.csv")
    return feat


if __name__ == "__main__":
    df = build_features()
    print(df.tail())
