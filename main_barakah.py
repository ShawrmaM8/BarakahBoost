from __future__ import annotations
import os
import datetime as dt
import pandas as pd
import streamlit as st
from utils.io_utils import ROOT, ensure_dirs, append_daily_log, load_json, save_json, daily_logs_df
from utils.validation import validate_prayers, clean_outcomes
from utils.scoring import weighted_baraka_score
from scripts.calculate_baraka import compute_scores
from scripts.train_model import train_and_analyze

# Initialize app configuration
st.set_page_config(
    page_title="Baraka Tracker",
    page_icon="✨",
    layout="wide"
)

# Ensure directories exist
ensure_dirs()

# Define paths
CFG_PATH = os.path.join(ROOT, "config", "config.json")
LOG_PATH = os.path.join(ROOT, "data", "raw", "daily_logs.json")

# Initialize session state
if "qrecs" not in st.session_state:
    st.session_state.qrecs = []


# ---- Helper Functions ----
def get_cfg() -> dict:
    """Load configuration from file."""
    return load_json(CFG_PATH, {})


def set_cfg(cfg: dict) -> None:
    """Save configuration to file."""
    save_json(CFG_PATH, cfg)


def render_log_day_page() -> None:
    """Render the daily logging page."""
    st.header("Log Today")

    with st.form("log_form", clear_on_submit=False):
        date = st.date_input("Date", dt.date.today()).isoformat()

        # Prayer on-time toggles
        st.subheader("Prayers")
        prayer_cols = st.columns(5)
        prayers = ["Fajr", "Dhuhr", "Asr", "Maghrib", "Isha"]
        prayer_values = {}

        for i, prayer in enumerate(prayers):
            with prayer_cols[i]:
                prayer_values[prayer.lower()] = st.checkbox(f"{prayer} on-time")

        # Quran recitation
        st.subheader("Qur'an Recitation")
        quran_recs = []
        q_cols = st.columns(3)

        with q_cols[0]:
            surah = st.text_input("Surah name (exact, e.g., Al-Kahf)")
        with q_cols[1]:
            ayahs = st.number_input("Ayahs read today (leave 0 for full surah)", 0, 600, 0)
        with q_cols[2]:
            if st.button("Add Recitation", key="add_recitation"):
                if surah:
                    quran_recs.append({"surah": surah, "ayahs": int(ayahs)})

        # Update session state
        if quran_recs:
            st.session_state.qrecs += quran_recs

        # Display current recitations
        if st.session_state.qrecs:
            st.table(pd.DataFrame(st.session_state.qrecs))

        if st.button("Clear Recitations", key="clear_recitations"):
            st.session_state.qrecs = []

        # Dhikr & Sadaqah
        st.subheader("Dhikr & Sadaqah")
        dhikr_reps = st.number_input("Total dhikr repetitions (all adhkar)", 0, 5000, 0)
        sadaqah_amount = st.number_input("Sadaqah given today (currency agnostic)",
                                         0.0, 100000.0, 0.0, step=0.5)

        # Sleep
        st.subheader("Sleep")
        sleep_cols = st.columns(2)
        with sleep_cols[0]:
            sleep_hours = st.number_input("Sleep duration (hours)", 0.0, 14.0, 0.0, step=0.25)
        with sleep_cols[1]:
            bedtime = st.text_input("Bedtime (24h HH:MM, optional)",
                                    placeholder="e.g., 22:30")

        # Screen Time
        st.subheader("Screen Time (minutes by app)")
        st.caption(
            "Tip: paste a few key apps and minutes for today; the app will categorize them using Settings → Screen Time lists.")

        app_minutes = {}
        for i in range(5):
            c1, c2 = st.columns([2, 1])
            with c1:
                app = st.text_input(f"App {i + 1}", key=f"app_{i}")
            with c2:
                mins = st.number_input(f"Minutes {i + 1}", 0.0, 1440.0, 0.0, key=f"mins_{i}")
            if app and mins:
                app_minutes[app] = float(mins)

        # Other Habits
        st.subheader("Other Habits")
        other_good = st.number_input("# of other good habits done", 0, 20, 0)
        other_bad = st.number_input("# of other bad habits occurred", 0, 20, 0)

        # Outcomes
        st.subheader("Outcomes (1-5)")
        clarity = st.slider("Clarity", 1, 5, 3)
        focus = st.slider("Focus", 1, 5, 3)
        calm = st.slider("Calm", 1, 5, 3)
        productivity = st.slider("Productivity", 1, 5, 3)

        submitted = st.form_submit_button("Save Day")

        if submitted:
            entry = {
                "date": date,
                "Fajr": prayer_values.get("fajr", False),
                "Dhuhr": prayer_values.get("dhuhr", False),
                "Asr": prayer_values.get("asr", False),
                "Maghrib": prayer_values.get("maghrib", False),
                "Isha": prayer_values.get("isha", False),
                "quran_recs": st.session_state.qrecs.copy(),
                "dhikr_reps": int(dhikr_reps),
                "sadaqah_amount": float(sadaqah_amount),
                "sleep_hours": float(sleep_hours),
                "bedtime": bedtime,
                "app_minutes": app_minutes,
                "other_good": int(other_good),
                "other_bad": int(other_bad),
                "clarity": int(clarity),
                "focus": int(focus),
                "calm": int(calm),
                "productivity": int(productivity)
            }

            try:
                append_daily_log(entry, LOG_PATH)
                st.success("Saved! You can switch to Dashboard to see updates.")
                # Clear recitations after successful submission
                st.session_state.qrecs = []
            except Exception as e:
                st.error(f"Error saving entry: {e}")


def render_dashboard_page() -> None:
    """Render the dashboard page."""
    st.header("Dashboard")

    try:
        scores = compute_scores()

        if scores.empty:
            st.info("No data yet. Log your first day.")
        else:
            # Baraka Score Chart
            st.subheader("Baraka Score (Daily)")
            st.line_chart(scores.set_index("date")["baraka_score"])

            # Components Chart
            st.subheader("Components")
            comp_cols = [c for c in scores.columns if c not in
                         ["date", "clarity", "focus", "calm", "productivity", "baraka_score"]]
            st.area_chart(scores.set_index("date")[comp_cols])

            # Outcomes Chart
            st.subheader("Outcomes")
            st.line_chart(scores.set_index("date")[["clarity", "focus", "calm", "productivity"]])

            # Latest scores summary
            st.subheader("Latest Scores")
            latest = scores.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Baraka Score", f"{latest['baraka_score']:.1f}")
            col2.metric("Prayer", f"{latest.get('prayer_on_time', 0):.1f}%")
            col3.metric("Quran", f"{latest.get('quran_recitation', 0):.1f}%")
            col4.metric("Dhikr", f"{latest.get('dhikr', 0):.1f}%")

    except Exception as e:
        st.error(f"Error loading dashboard: {e}")


def render_insights_page() -> None:
    """Render the insights page."""
    st.header("Insights & Personalization")

    try:
        res = train_and_analyze()

        if res.get("status") in ("no_data", "insufficient_data"):
            st.info("Need more days with outcomes to compute robust insights.")
        else:
            # Model performance
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            col1.metric("CV R² (mean)", f"{res['cv_r2_mean']:.3f}")
            col2.metric("CV R² (std)", f"{res['cv_r2_std']:.3f}")

            # Feature importances
            st.subheader("Feature Importances")
            fi = pd.DataFrame.from_dict(res["feature_importances"],
                                        orient="index", columns=["importance"])
            fi = fi.sort_values("importance", ascending=False)
            st.bar_chart(fi)

            # Correlations
            st.subheader("Correlations with Outcome")
            corr = pd.DataFrame.from_dict(res["correlations_with_outcome"],
                                          orient="index", columns=["correlation"])
            st.bar_chart(corr)

            # Personalized tips
            st.subheader("Actionable Suggestions")
            scores = compute_scores()

            if not scores.empty:
                latest = scores.iloc[-1]
                tips = []

                if latest.get("prayer_on_time", 0) < 80:
                    tips.append("Aim to pray all five on time tomorrow.")
                if latest.get("screen_time", 100) < 60:
                    tips.append("Reduce distracting apps by ~20 minutes; shift that to Qur'an or reading.")
                if latest.get("quran_recitation", 0) < 40:
                    tips.append("Add one short surah after Fajr (e.g., Al-Ikhlas/Al-Falaq/An-Nas).")
                if latest.get("sleep", 0) < 70:
                    tips.append("Target 7-8.5 hours and lights out before 23:00.")
                if latest.get("dhikr", 0) < 30:
                    tips.append("Sprinkle 100 dhikr reps across the day (commute, waiting time).")

                if tips:
                    for t in tips:
                        st.write("•", t)
                else:
                    st.success("Great momentum—keep consistent!")

    except Exception as e:
        st.error(f"Error generating insights: {e}")


def render_settings_page() -> None:
    """Render the settings page."""
    st.header("Settings")

    try:
        cfg = get_cfg()

        # Weights settings
        st.subheader("Weights")
        for k in list(cfg["weights"].keys()):
            cfg["weights"][k] = st.slider(k, 0.0, 1.0, float(cfg["weights"][k]), 0.01)

        # Screen time categorization
        st.subheader("Screen Time Categorization")
        pa = st.text_area("Productive apps (comma-separated)",
                          ", ".join(cfg["screen_time"]["productive_apps"]))
        da = st.text_area("Distracting apps (comma-separated)",
                          ", ".join(cfg["screen_time"]["distracting_apps"]))

        cfg["screen_time"]["productive_apps"] = [x.strip() for x in pa.split(",") if x.strip()]
        cfg["screen_time"]["distracting_apps"] = [x.strip() for x in da.split(",") if x.strip()]

        # Sleep settings
        st.subheader("Sleep Settings")
        c1, c2 = st.columns(2)
        cfg["sleep"]["ideal_min_hours"] = c1.number_input("Ideal min hours", 4.0, 10.0,
                                                          float(cfg["sleep"]["ideal_min_hours"]), 0.25)
        cfg["sleep"]["ideal_max_hours"] = c2.number_input("Ideal max hours", 5.0, 12.0,
                                                          float(cfg["sleep"]["ideal_max_hours"]), 0.25)
        cfg["sleep"]["bedtime_bonus_before"] = st.text_input("Bedtime bonus before (HH:MM)",
                                                             cfg["sleep"]["bedtime_bonus_before"])

        if st.button("Save Settings"):
            set_cfg(cfg)
            st.success("Settings saved.")

    except Exception as e:
        st.error(f"Error loading settings: {e}")


def render_data_page() -> None:
    """Render the data page."""
    st.header("Raw & Processed Data")

    try:
        # Raw logs
        st.subheader("Raw Logs")
        logs = daily_logs_df()
        st.dataframe(logs if not logs.empty else pd.DataFrame())

        # Processed data
        st.subheader("Processed Features & Scores")
        scores = compute_scores()
        st.dataframe(scores if not scores.empty else pd.DataFrame())

        # Data summary
        if not logs.empty:
            st.subheader("Data Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Days", len(logs))
            col2.metric("Days with Outcomes", len(scores) if not scores.empty else 0)
            col3.metric("First Entry", logs["date"].min())

    except Exception as e:
        st.error(f"Error loading data: {e}")


# ---- Main App ----
def main():
    """Main application function."""
    # Sidebar Navigation
    st.sidebar.title("✨ Baraka Tracker")
    page = st.sidebar.radio("Go to", ["Log Day", "Dashboard", "Insights", "Settings", "Data"])

    # Render the selected page
    if page == "Log Day":
        render_log_day_page()
    elif page == "Dashboard":
        render_dashboard_page()
    elif page == "Insights":
        render_insights_page()
    elif page == "Settings":
        render_settings_page()
    elif page == "Data":
        render_data_page()


if __name__ == "__main__":
    main()