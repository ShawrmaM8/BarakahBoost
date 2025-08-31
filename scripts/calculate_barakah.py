from __future__ import annotations
import pandas as pd
from typing import Dict, Any
from utils.io_utils import ROOT, ensure_dirs, read_config, save_df_csv
from utils.scoring import (
    score_prayer, score_quran, score_dhikr, score_sadaqah, score_sleep,
    score_screen_time, score_other, weighted_baraka_score
)
from scripts.process_data import build_features


def compute_scores() -> pd.DataFrame:
    """
    Compute daily Baraka scores based on various spiritual activities and metrics.

    Returns:
        pd.DataFrame: DataFrame containing daily scores and metrics
    """
    # Ensure necessary directories exist
    ensure_dirs()

    # Read configuration
    cfg = read_config()

    # Build features from raw data
    feat = build_features()

    # Return empty DataFrame if no features available
    if feat.empty:
        return pd.DataFrame()

    rows = []
    for _, row in feat.iterrows():
        # Calculate score components
        components = calculate_score_components(row, cfg)

        # Calculate overall weighted Baraka score
        score = weighted_baraka_score(components, cfg["weights"])

        # Append results
        rows.append({
            "date": row["date"],
            **components,
            "baraka_score": score,
            "clarity": row.get("clarity"),
            "focus": row.get("focus"),
            "calm": row.get("calm"),
            "productivity": row.get("productivity")
        })

    # Create output DataFrame and sort by date
    out = pd.DataFrame(rows).sort_values("date")

    # Save results
    save_df_csv(out, "data/processed/baraka_scores.csv")

    # Save outcomes separately
    outcomes = out[["date", "clarity", "focus", "calm", "productivity"]]
    save_df_csv(outcomes, "data/processed/outcomes.csv")

    return out


def calculate_score_components(row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate individual score components for Baraka calculation.

    Args:
        row: Row from features DataFrame
        cfg: Configuration dictionary

    Returns:
        Dictionary containing all score components
    """
    components = {}

    # Prayer score
    components["prayer_on_time"] = row["prayer_on_time"] * 100.0

    # Quran recitation score
    quran_max = cfg["quran"]["max_daily_points"]
    quran_items = row.get("quran_items", 0)
    components["quran_recitation"] = min(100.0, (quran_items / quran_max) * 100.0)

    # Dhikr score
    dhikr_config = cfg["dhikr"]
    dhikr_reps = row.get("dhikr_reps", 0)
    dhikr_points = dhikr_reps * dhikr_config["points_per_repetition"]
    components["dhikr"] = min(100.0, (dhikr_points / dhikr_config["max_daily_points"]) * 100.0)

    # Sadaqah score
    sadaqah_amount = float(row.get("sadaqah_amount", 0.0))
    sadaqah_config = cfg["sadaqah"]
    components["sadaqah"] = score_sadaqah(
        sadaqah_amount,
        sadaqah_config["log_scale"],
        sadaqah_config["log_base"],
        sadaqah_config["max_daily_points"]
    )

    # Sleep score
    sleep_hours = float(row.get("sleep_hours", 0.0))
    bedtime = str(row.get("bedtime", "")) or None
    sleep_config = cfg["sleep"]
    components["sleep"] = score_sleep(
        sleep_hours,
        bedtime,
        sleep_config["ideal_min_hours"],
        sleep_config["ideal_max_hours"],
        sleep_config["bedtime_bonus_before"],
        sleep_config["max_daily_points"]
    )

    # Screen time score
    screen_time_data = {
        "productive": row.get("prod_minutes", 0.0),
        "distracting": row.get("dist_minutes", 0.0)
    }
    screen_time_config = {
        "productive_apps": ["productive"],
        "distracting_apps": ["distracting"],
        "max_daily_minutes": cfg["screen_time"]["max_daily_minutes"]
    }
    components["screen_time"] = score_screen_time(screen_time_data, screen_time_config)

    # Other activities scores
    other_config = cfg["other"]
    other_good = int(row.get("other_good", 0))
    other_bad = int(row.get("other_bad", 0))
    components["other_good"] = score_other(
        other_good, 0,
        other_config["good_points"],
        other_config["bad_points"]
    )
    components["other_bad"] = score_other(
        0, other_bad,
        other_config["good_points"],
        other_config["bad_points"]
    )

    return components


if __name__ == "__main__":
    try:
        df = compute_scores()
        print("Baraka scores calculated successfully!")
        print("\nLatest scores:")
        print(df.tail())
    except Exception as e:
        print(f"Error calculating Baraka scores: {e}")
        # Optionally, you could log this error to a file