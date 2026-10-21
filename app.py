"""Streamlit dashboard: ATP match win probabilities (historical demo).

Run from the project root:   streamlit run app.py
Needs the local artifact:    python -m src.build_artifacts
All numbers come from src/predictor.py (trained models + the research feature pipeline); nothing is simulated.
"""
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.evaluate import read_md_table
from src.predictor import ARTIFACT, NEW, SURFACES, Predictor

os.chdir(Path(__file__).parent)  # data and report paths are relative to the project root, wherever streamlit is started
st.set_page_config(page_title="ATP match win probabilities", layout="wide")
GROUP_LABELS = {"elo": "Elo ratings", "rank": "Ranking", "last": "Last 5/10", "win": "Win-rate windows", "surf": "Surface form",
                "rest": "Rest/workload", "dec": "Decayed form", "adj": "Opp-adjusted", "adjdec": "Opp-adj. decayed"}
SHOWN = {"A Elo": "Elo (benchmark)", "C LR": "Model C (challenger)", "D LR": "Model D (primary)"}


@st.cache_resource
def load():
    return Predictor.load()


if not ARTIFACT.exists():
    st.error("The model artifact is missing. Build it first:  `python -m src.build_artifacts`")
    st.stop()
P = load()
art = P.art

st.title("ATP match win probabilities")
st.warning(f"**Historical demo, not a current forecast.** The data ends with tournaments starting **{art['data_end']}**. Every prediction "
           f"describes a hypothetical match on the snapshot date **{art['snapshot_date']}**, using only results available by then. "
           "The models know nothing after that date. This is a research tool, not betting advice.")

with st.sidebar:
    st.header("Match")
    inactive = st.checkbox("Include inactive players", help="Players with no match in the year before the snapshot. Their ratings are stale.")
    players = P.players(active_only=not inactive)
    options = list(players.index) + list(NEW)

    def label(pid):
        if pid in NEW:
            return f"New player (no history) {pid[-1]}"
        r = players.loc[pid]
        return f"{r['name']} (rank {int(r['rank'])})" if pd.notna(r["rank"]) else f"{r['name']} (unranked)"

    a_id = st.selectbox("Player A", options, index=0, format_func=label)
    b_id = st.selectbox("Player B", options, index=1, format_func=label)
    surface = st.selectbox("Surface", SURFACES)
    st.caption(f"{len(players):,} players listed. Source: {art['source']}.")

tab_pred, tab_back, tab_about = st.tabs(["Prediction", "Backtest results", "About and limits"])

with tab_pred:
    if a_id == b_id:
        st.error("Choose two different players.")
    else:
        r = P.predict(a_id, b_id, surface)
        A, B = r["A"], r["B"]
        st.subheader(f"{A['name']} vs {B['name']} on {surface}")
        cols = st.columns(3)
        for col, key, title in zip(cols, ("D", "C", "Elo"), ("Model D (primary)", "Model C (challenger)", "Elo (benchmark)")):
            col.metric(title, f"{r['p'][key]:.1%}")
            col.caption(f"{A['name']} wins {r['p'][key]:.1%}  \n{B['name']} wins {1 - r['p'][key]:.1%}")
        if not r["reliable"]:
            st.error("**Not a reliable prediction.** An input needed for an honest estimate is missing or too old (see below). The number is shown for transparency only.")
        for note in r["notes"]:
            st.info(note)

        st.markdown("#### What the models see")
        fa, fb = A["features"], B["features"]
        rank = lambda p: "none" if np.isnan(p["rank"]) else f"{int(p['rank'])} (as of {p['last_date']})"
        rows = [("General Elo", f"{A['elo']:.0f}", f"{B['elo']:.0f}"),
                (f"{surface} Elo", f"{A['surface_elo']:.0f}", f"{B['surface_elo']:.0f}"),
                ("Last known ranking", rank(A), rank(B)),
                ("Ranking points", *("none" if np.isnan(p["points"]) else f"{int(p['points']):,}" for p in (A, B))),
                ("Matches in the data", f"{A['matches']:,}", f"{B['matches']:,}"),
                ("Win rate, last 10 matches*", f"{fa['wr10']:.0%}", f"{fb['wr10']:.0%}"),
                ("Win rate, last 365 days*", f"{fa['wr365']:.0%}", f"{fb['wr365']:.0%}"),
                (f"Win rate on {surface.lower()}, last 365 days*", f"{fa['swr365']:.0%}", f"{fb['swr365']:.0%}"),
                ("Matches in the last 90 days", f"{np.expm1(fa['ln90']):.0f}", f"{np.expm1(fb['ln90']):.0f}"),
                ("Days since last tournament counted as finished", *("n/a" if p["days_since_release"] is None else str(p["days_since_release"]) for p in (A, B))),
                ("Opponent-adjusted form, last 10**", f"{fa['adj10']:+.3f}", f"{fb['adj10']:+.3f}")]
        st.dataframe(pd.DataFrame(rows, columns=["Input", A["name"], B["name"]]), hide_index=True, width="stretch")
        st.caption("*Rates are pulled toward 50% by a prior worth 3 matches, so a player with few matches is not judged on them alone. "
                   "**Average of (result minus the Elo-expected result): positive means the player beat expectations, so wins over strong opponents count more. "
                   "A tournament counts as finished only after its maximum possible length, because the data has no match dates.")

        st.markdown("#### What moves Model D")
        c = pd.Series(r["contributions"]["D"]).rename(GROUP_LABELS).sort_values(ascending=False).rename("Contribution").rename_axis("Feature group").reset_index()
        st.altair_chart(alt.Chart(c).mark_bar(color="#2a78d6", cornerRadiusEnd=4, size=18).encode(
            x=alt.X("Contribution:Q", title=f"Push toward {A['name']} (positive) or {B['name']} (negative), in log-odds"),
            y=alt.Y("Feature group:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
            tooltip=["Feature group", alt.Tooltip("Contribution:Q", format="+.3f")]).properties(height=300), width="stretch")
        st.caption("Each bar is one feature group's push on the prediction. Groups overlap heavily (form and Elo carry similar information), "
                   "so opposite-signed bars often cancel: read the total, not single bars.")

with tab_back:
    st.markdown("These are the results of the **same model specifications**, frozen before the test data was opened and scored **once** "
                "on matches the models had never seen. The models in this dashboard are those specifications refitted on all data, "
                "which cannot itself be tested.")
    text = Path("reports/phase6_backtest.md").read_text()
    gains = {}
    for year, title in (("2025", "2025: independent test (full season)"), ("2026", "2026: forward test, **January to Roland Garros only**")):
        t = read_md_table(text, f"### {year}.1")
        t = t[t.model.isin(SHOWN)].assign(model=lambda d: d.model.map(SHOWN))
        st.markdown(f"#### {title}")
        st.dataframe(t.rename(columns={"model": "Model", "n": "Matches", "log_loss": "Log loss", "95% CI": "95% interval", "brier": "Brier", "accuracy": "Accuracy",
                                       "ece": "ECE", "cal. intercept": "Calibration intercept", "cal. slope": "Calibration slope"}), hide_index=True, width="stretch")
        ll = t.set_index("model").log_loss.astype(float)
        gains[year if year == "2025" else "2026 (Jan-May)"] = (ll["Elo (benchmark)"] - ll).drop("Elo (benchmark)")
        d = read_md_table(text, f"### {year}.2")
        d = d[d.comparison.isin(["D LR minus A Elo", "C LR minus A Elo", "D LR minus C LR"])]
        d = d.assign(comparison=d.comparison.str.replace("D LR", "Model D").str.replace("C LR", "Model C").str.replace("A Elo", "Elo"))
        st.dataframe(d.rename(columns={"tier": "Tier", "comparison": "Comparison (log loss)", "diff": "Difference", "low": "Low", "high": "High",
                                       "first clearly better?": "First clearly better?"}), hide_index=True, width="stretch")
    st.caption("Lower log loss is better. Differences are the first model minus the second, with 95% intervals from resampling whole tournaments. "
               "Model D beats Elo in both periods; Model C matches Model D.")
    st.markdown("#### Improvement over Elo (log loss)")
    g = pd.DataFrame(gains).rename_axis("Model").reset_index().melt("Model", var_name="Period", value_name="Improvement")
    st.altair_chart(alt.Chart(g).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=240)), xOffset="Period:N",
        y=alt.Y("Improvement:Q", title="Elo log loss minus model log loss (higher is better)"),
        color=alt.Color("Period:N", scale=alt.Scale(range=["#2a78d6", "#eb6834"]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["Model", "Period", alt.Tooltip("Improvement:Q", format=".4f")]).properties(height=320), width="stretch")
    st.markdown("#### Calibration")
    c1, c2 = st.columns(2)
    c1.image("reports/figures/phase6_calibration_2025.png", caption="2025. Points on the dashed line mean the stated probability matched reality.")
    c2.image("reports/figures/phase6_calibration_2026.png", caption="2026, January to Roland Garros (1,310 matches: noisier).")
    st.caption("Elo is overconfident (calibration slope about 0.8). Model D is closer (about 0.92 to 0.95) but still slightly overconfident. "
               "No recalibration was applied, because the only data left to fit it on is the already-used test data. "
               "Full report, including the other research models: reports/phase6_backtest.md.")

with tab_about:
    st.markdown(f"""
**Question.** How well can pre-match information predict who wins an ATP match, and do recent form and opponent strength add anything beyond an Elo rating?

**Data.** {art['n_matches']:,} ATP matches from {art['data_start']} to tournaments starting {art['data_end']}, compiled by {art['source']}.
Non-commercial use only. The raw data is not distributed with this project.

**Models.**
- **Elo (benchmark):** one rating per player, updated after every result.
- **Model C (challenger):** Logistic Regression on Elo, surface Elo, ranking and recent form.
- **Model D (primary):** Model C plus opponent-adjusted form.

Both models were refitted on {art['n_train']:,} tour-level matches from 2015 onward, with settings frozen before the independent test.

**Limits you should know.**
- **Not current.** Predictions are for {art['snapshot_date']}. Rankings are each player's last ranking in the data.
- **No match dates in the source.** A tournament's results are used only after its maximum possible length, so the newest form arrives late.
- **Inactivity is not modelled.** Ratings do not fade while a player is away; long-inactive players are hidden by default and flagged.
- **Context is ignored.** Best-of-five, injuries, travel, altitude, indoor or outdoor and head-to-head are not inputs.
- **Modest edge.** In the 2025 test Model D picked the winner in 65.1% of matches against 63.6% for Elo.
- **No market comparison.** Betting odds were never used, as inputs or as a benchmark. This is not betting advice.
""")
