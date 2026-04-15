import json
import os
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
from dotenv import load_dotenv

from simulate import simulate_match
from utils.prediction_storage import save_prediction
from utils.data_updater import append_new_matches
from main import get_cached_fixture, clear_outdated_cache

load_dotenv()
API_KEY = os.getenv("FOOTBALL_API_KEY")

TEAMS_FILE = os.path.join("config", "teams.json")
PRED_FILE = "data/predictions.json"

MODEL_LABELS = {
    "poisson": "Statistical — Poisson",
    "random_forest": "AI — Random Forest",
    "xgboost": "AI — XGBoost",
}

MODEL_ACCURACY = {
    "poisson": "~46%",
    "random_forest": "~49%",
    "xgboost": "~50%",
}

CONF_COLORS = {
    "High": "#2e7d32",
    "Medium": "#f9a825",
    "Low": "#c62828",
}


# ── data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    append_new_matches()
    return pd.read_csv("data/historical_matches.csv")


@st.cache_data(show_spinner=False)
def load_csv():
    """lightweight read for form/h2h — no network call"""
    if not os.path.exists("data/historical_matches.csv"):
        return pd.DataFrame()
    df = pd.read_csv("data/historical_matches.csv", dtype=str)
    df = df[df["homeGoals"].notna() & (df["homeGoals"] != "")]
    df["date_parsed"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    return df


def load_teams():
    with open(TEAMS_FILE) as f:
        teams = json.load(f)
    return dict(sorted(teams.items(), key=lambda x: x[1]))


def load_predictions():
    if not os.path.exists(PRED_FILE):
        return []
    with open(PRED_FILE) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


# ── stat helpers ──────────────────────────────────────────────────────────────

def get_recent_form(df, team, n=5):
    matches = df[(df["homeTeam"] == team) | (df["awayTeam"] == team)]
    matches = matches.sort_values("date_parsed", ascending=False).head(n)
    results = []
    for _, row in matches.iterrows():
        hg, ag = int(float(row["homeGoals"])), int(float(row["awayGoals"]))
        if row["homeTeam"] == team:
            results.append("W" if hg > ag else ("D" if hg == ag else "L"))
        else:
            results.append("W" if ag > hg else ("D" if ag == hg else "L"))
    return results  # most recent first


def form_html(results):
    colors = {"W": "#2e7d32", "D": "#757575", "L": "#c62828"}
    badges = ""
    for r in results:
        c = colors.get(r, "#999")
        badges += (
            f'<span style="display:inline-block;width:22px;height:22px;border-radius:4px;'
            f'background:{c};color:white;text-align:center;line-height:22px;'
            f'font-size:0.75rem;font-weight:700;margin-right:3px;">{r}</span>'
        )
    return badges or "<span style='opacity:0.4;font-size:0.8rem;'>No data</span>"


def get_avg_goals(df, team, n=10):
    matches = df[(df["homeTeam"] == team) | (df["awayTeam"] == team)]
    matches = matches.sort_values("date_parsed", ascending=False).head(n)
    scored, conceded = [], []
    for _, row in matches.iterrows():
        hg, ag = float(row["homeGoals"]), float(row["awayGoals"])
        if row["homeTeam"] == team:
            scored.append(hg);
            conceded.append(ag)
        else:
            scored.append(ag);
            conceded.append(hg)
    if not scored:
        return 0.0, 0.0
    return sum(scored) / len(scored), sum(conceded) / len(conceded)


def get_h2h(df, home, away, n=5):
    h2h = df[
        ((df["homeTeam"] == home) & (df["awayTeam"] == away)) |
        ((df["homeTeam"] == away) & (df["awayTeam"] == home))
        ]
    return h2h.sort_values("date_parsed", ascending=False).head(n)


# ── ui helpers ────────────────────────────────────────────────────────────────

def confidence_label(probs):
    best = max(probs.values())
    if best >= 0.60:
        return "High"
    elif best >= 0.45:
        return "Medium"
    return "Low"


def confidence_badge(label):
    color = CONF_COLORS[label]
    st.markdown(
        f"""
        <div style="text-align:center;">
            <div style="font-size:0.875rem; font-weight:400; margin-bottom:6px;">Confidence</div>
            <div style="display:inline-block; background:{color};
                        padding:4px 18px; border-radius:6px; font-weight:700; font-size:1.1rem;">
                {label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prob_bar(label, value, color):
    st.markdown(
        f"""
        <div style="margin-bottom:6px;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:2px;">
                <span>{label}</span><span>{value:.0%}</span>
            </div>
            <div style="background:#e0e0e0; border-radius:4px; height:10px;">
                <div style="width:{value * 100:.1f}%; background:{color}; border-radius:4px; height:10px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_outcome(col):
    styles = []
    for val in col:
        if val == "Correct":
            styles.append("background-color: #2e7d32; color: white; font-weight: 700;")
        elif val == "Incorrect":
            styles.append("background-color: #c62828; color: white; font-weight: 700;")
        elif val == "Pending":
            styles.append("background-color: #f9a825; color: black; font-weight: 700;")
        else:
            styles.append("")
    return styles


# ── page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Football Predictor", layout="centered")
st.title("Football Predictor")
st.markdown(
    """
    <div style="font-size: 0.85rem; opacity: 0.75;">
    A data-driven application developed as a final year college project for predicting
    Premier League match outcomes using statistical and machine learning models.
    <br><br>
    <b>How to use:</b> Select a team to load their next fixture, choose a prediction model, then
    hit <b>Run Prediction</b> to see win probabilities and the most likely scoreline.
    Use <b>Compare All Models</b> to see Poisson, Random Forest and XGBoost side by side.
    Past predictions and their accuracy are tracked in the <b>History</b> tab.
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    [data-testid="stMetric"] { text-align: center; }
    [data-testid="stMetricValue"] { justify-content: center; display: flex; }
    [data-testid="stMetricLabel"] { justify-content: center; display: flex; width: 100%; }
    [data-testid="stMetricLabel"] p { text-align: center; width: 100%; }
    hr { margin-top: 0.8rem !important; margin-bottom: 0.8rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_history = st.tabs(["Predict", "History"])

# ── predict tab ───────────────────────────────────────────────────────────────
with tab_predict:
    teams = load_teams()
    team_names = list(teams.values())
    team_ids = list(teams.keys())

    selected_name = st.selectbox("Select a team", team_names)
    selected_id = int(team_ids[team_names.index(selected_name)])

    clear_outdated_cache()
    fixture = get_cached_fixture(selected_id, API_KEY)

    if not fixture:
        st.warning("No upcoming fixture found for this team.")
        st.stop()

    fixture_dt = datetime.fromisoformat(fixture["date"].replace("Z", "+00:00"))
    fixture_str = fixture_dt.strftime("%A, %d %b %Y  \u00b7  %H:%M UTC")

    st.markdown("---")
    st.markdown("#### Upcoming Fixture")
    st.markdown(
        f"""
        <div style="text-align: center; padding: 12px 0;">
            <span style="font-size: 1.4rem; font-weight: 700;">{fixture['home']}</span>
            &nbsp;&nbsp;
            <span style="font-size: 1.2rem; opacity: 0.6;">vs</span>
            &nbsp;&nbsp;
            <span style="font-size: 1.4rem; font-weight: 700;">{fixture['away']}</span>
            <br/>
            <span style="font-size: 0.85rem; opacity: 0.6;">{fixture_str}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # form + goals per game
    csv_df = load_csv()
    if not csv_df.empty:
        home_form = get_recent_form(csv_df, fixture["home"])
        away_form = get_recent_form(csv_df, fixture["away"])
        home_scored, home_conc = get_avg_goals(csv_df, fixture["home"])
        away_scored, away_conc = get_avg_goals(csv_df, fixture["away"])

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown(
                f"""
                <div style="font-size:0.78rem; opacity:0.55; margin-bottom:4px;">
                    {fixture['home']} · last 5
                </div>
                <div style="margin-bottom:4px;">{form_html(home_form)}</div>
                <div style="font-size:0.78rem; opacity:0.55;">
                    Scored {home_scored:.1f} &nbsp;·&nbsp; Conceded {home_conc:.1f} per game
                </div>
                """,
                unsafe_allow_html=True,
            )
        with fc2:
            st.markdown(
                f"""
                <div style="font-size:0.78rem; opacity:0.55; margin-bottom:4px; text-align:right;">
                    {fixture['away']} · last 5
                </div>
                <div style="text-align:right; margin-bottom:4px;">{form_html(away_form)}</div>
                <div style="font-size:0.78rem; opacity:0.55; text-align:right;">
                    Scored {away_scored:.1f} &nbsp;·&nbsp; Conceded {away_conc:.1f} per game
                </div>
                """,
                unsafe_allow_html=True,
            )

        # h2h expander
        st.markdown("<div style='margin-top:5px;'>", unsafe_allow_html=True)
        h2h = get_h2h(csv_df, fixture["home"], fixture["away"])
        with st.expander(f"Head-to-head · last {len(h2h)} meetings"):
            if h2h.empty:
                st.caption("No previous meetings in the dataset.")
            else:
                for _, row in h2h.iterrows():
                    hg, ag = int(float(row["homeGoals"])), int(float(row["awayGoals"]))
                    date_str = row["date_parsed"].strftime("%d %b %Y")
                    st.markdown(
                        f"<div style='font-size:0.85rem; padding:3px 0;'>"
                        f"{row['homeTeam']} <b>{hg}–{ag}</b> {row['awayTeam']}"
                        f"&nbsp;&nbsp;<span style='opacity:0.5;'>{date_str}</span></div>",
                        unsafe_allow_html=True,
                    )
            st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    model_label = st.radio(
        "Prediction model",
        list(MODEL_LABELS.values()),
        horizontal=True,
    )
    model_key = [k for k, v in MODEL_LABELS.items() if v == model_label][0]
    st.caption(f"Evaluated accuracy: {MODEL_ACCURACY[model_key]} on holdout test set")

    col_single, col_compare = st.columns(2)
    run_single = col_single.button("Run Prediction", type="primary", width="stretch")
    run_compare = col_compare.button("Compare All Models", width="stretch")

    # single model prediction
    if run_single:
        with st.spinner("Running prediction..."):
            df = load_data()
            result = simulate_match(
                model_name=model_key,
                home_team=fixture["home"],
                away_team=fixture["away"],
                df=df,
            )

        probs = result["probabilities"]
        top_score = result["top_score"]
        top_pct = result["top_score_percentage"]
        conf_label = confidence_label(probs)

        st.subheader("Result")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{fixture['home']} win", f"{probs['home_win']:.0%}")
        c2.metric("Draw", f"{probs['draw']:.0%}")
        c3.metric(f"{fixture['away']} win", f"{probs['away_win']:.0%}")
        with c4:
            confidence_badge(conf_label)

        st.markdown("<br>", unsafe_allow_html=True)
        prob_bar(f"{fixture['home']} win", probs["home_win"], "#4C78A8")
        prob_bar("Draw", probs["draw"], "#9FA8B0")
        prob_bar(f"{fixture['away']} win", probs["away_win"], "#E45756")

        st.markdown("---")
        st.markdown(
            f"**Most likely scoreline:** &nbsp; `{fixture['home']}` &nbsp; **{top_score}** &nbsp; `{fixture['away']}`"
            f"&nbsp;&nbsp; *(probability {top_pct:.1%})*"
        )

        with st.expander("Top 5 scorelines"):
            dist = result["score_distribution"]
            total = sum(dist.values())
            for score, count in list(dist.items())[:5]:
                st.write(f"`{score}` — {count / total:.1%}")

        save_prediction(
            model_name=result["model_used"],
            home_team=fixture["home"],
            away_team=fixture["away"],
            top_score=top_score,
            fixture_date=fixture["date"],
        )
        st.success("Prediction saved to history.")

    # model comparison
    if run_compare:
        with st.spinner("Running all models..."):
            df = load_data()
            results = {
                k: simulate_match(model_name=k, home_team=fixture["home"], away_team=fixture["away"], df=df)
                for k in MODEL_LABELS
            }

        st.subheader("Model Comparison")

        cols = st.columns(3)
        for col, (key, label) in zip(cols, MODEL_LABELS.items()):
            r = results[key]
            p = r["probabilities"]
            conf = confidence_label(p)
            with col:
                st.markdown(
                    f"<div style='text-align:center; font-weight:700; margin-bottom:4px;'>{label}</div>"
                    f"<div style='text-align:center; font-size:0.75rem; opacity:0.55; margin-bottom:8px;'>"
                    f"accuracy {MODEL_ACCURACY[key]}</div>",
                    unsafe_allow_html=True,
                )
                st.metric(f"{fixture['home']} win", f"{p['home_win']:.0%}")
                st.metric("Draw", f"{p['draw']:.0%}")
                st.metric(f"{fixture['away']} win", f"{p['away_win']:.0%}")
                st.markdown(
                    f"<div style='text-align:center; font-size:0.85rem; margin-bottom:20px;'>"
                    f"Top score: {r['top_score']}</div>",
                    unsafe_allow_html=True,
                )
                confidence_badge(conf)

# ── history tab ───────────────────────────────────────────────────────────────
with tab_history:
    predictions = load_predictions()

    if not predictions:
        st.info("No predictions saved yet.")
    else:
        df_hist = pd.read_csv("data/historical_matches.csv", dtype=str)

        # build all rows
        all_rows = []
        for p in reversed(predictions):
            model_display = MODEL_LABELS.get(p["model"], p["model"])
            fixture_dt = datetime.fromisoformat(p["fixture_date"].replace("Z", "+00:00"))
            pred_dt = datetime.fromisoformat(p["prediction_date"])
            fixture_day = fixture_dt.strftime("%d/%m/%Y")

            match = df_hist[
                (df_hist["homeTeam"] == p["home_team"]) &
                (df_hist["awayTeam"] == p["away_team"]) &
                (df_hist["date"].str.startswith(fixture_day))
                ]

            if not match.empty and match.iloc[0]["homeGoals"] not in ("", None):
                ah, aa = match.iloc[0]["homeGoals"], match.iloc[0]["awayGoals"]
                actual_score = f"{int(float(ah))}-{int(float(aa))}"
                ph, pa = map(int, p["predicted_score"].split("-"))
                ih, ia = int(float(ah)), int(float(aa))
                pred_outcome = "H" if ph > pa else ("D" if ph == pa else "A")
                act_outcome = "H" if ih > ia else ("D" if ih == ia else "A")
                outcome = "Correct" if pred_outcome == act_outcome else "Incorrect"
            else:
                actual_score = "—"
                outcome = "Pending"

            all_rows.append({
                "Home": p["home_team"],
                "Away": p["away_team"],
                "Fixture date": fixture_dt.strftime("%d %b %Y"),
                "Model": model_display,
                "Predicted": p["predicted_score"],
                "Actual": actual_score,
                "Outcome": outcome,
                "Predicted on": pred_dt.strftime("%d %b %Y %H:%M"),
            })

        # summary stats
        resolved = [r for r in all_rows if r["Outcome"] != "Pending"]
        correct = [r for r in resolved if r["Outcome"] == "Correct"]
        pct = len(correct) / len(resolved) * 100 if resolved else 0

        stats_col, clear_col = st.columns([4, 1])
        with stats_col:
            st.caption(
                f"{len(all_rows)} predictions · {len(resolved)} resolved · "
                f"{len(correct)} correct ({pct:.0f}%)"
            )
        with clear_col:
            st.markdown(
                """
                <style>
                .clear-marker + div[data-testid="stButton"] button {
                    background-color: #1976D2 !important;
                    color: white !important;
                    border: none !important;
                }
                .clear-marker + div[data-testid="stButton"] button:hover {
                    background-color: #1565C0 !important;
                }
                </style>
                <div class="clear-marker"></div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Clear history", width="stretch"):
                with open(PRED_FILE, "w") as f:
                    json.dump([], f)
                st.rerun()

        selected_models = st.multiselect(
            "Filter by model", list(MODEL_LABELS.values()), default=list(MODEL_LABELS.values())
        )

        filtered = [r for r in all_rows if r["Model"] in selected_models]

        if filtered:
            styled = pd.DataFrame(filtered).style.apply(style_outcome, subset=["Outcome"])
            st.dataframe(styled, width="stretch", hide_index=True)
        else:
            st.info("No predictions match the selected filters.")
