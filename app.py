"""
app.py — 2026 World Cup forecasting dashboard.

Loads the locked model artifacts + predictions and serves them interactively:
  • Title odds (Monte Carlo)            • Every group-stage match prediction
  • A live "predict any match" tool     • Methodology + honesty caveats

Run locally:   streamlit run app.py
Deploy:        push to GitHub -> share.streamlit.io -> point at this file.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from scipy.stats import poisson

DATA = Path(__file__).parent / "outputs"          # where the locked CSVs live

# ----------------------------------------------------------------------------
# Data loading — cached so it runs ONCE, not on every interaction
# ----------------------------------------------------------------------------
@st.cache_data
def load_model():
    S = pd.read_csv(DATA / "team_strengths.csv", index_col=0)
    meta = json.load(open(DATA / "_glm_meta.json"))
    return S["attack"].to_dict(), S["defend"].to_dict(), meta

@st.cache_data
def load_predictions():
    odds = pd.read_csv(DATA / "locked_title_odds.csv")
    games = pd.read_csv(DATA / "locked_group_predictions.csv")
    manifest = json.load(open(DATA / "lock_manifest.json"))
    return odds, games, manifest

att, dfn, META = load_model()
INT, HOME, RHO = META["intercept"], META["home"], META["rho"]
odds, games, manifest = load_predictions()
TEAMS = sorted(att.keys())

# ----------------------------------------------------------------------------
# The model — identical maths to the notebook, so app & analysis never diverge
# ----------------------------------------------------------------------------
def score_matrix(home, away, neutral, maxg=8):
    ish = 0 if neutral else 1
    lh = np.exp(INT + att[home] + dfn[away] + HOME * ish)
    la = np.exp(INT + att[away] + dfn[home])
    M = np.outer(poisson.pmf(range(maxg+1), lh), poisson.pmf(range(maxg+1), la))
    M[0,0]*=1-lh*la*RHO; M[0,1]*=1+lh*RHO; M[1,0]*=1+la*RHO; M[1,1]*=1-RHO
    return M / M.sum(), lh, la

def predict(home, away, neutral):
    M, lh, la = score_matrix(home, away, neutral)
    i, j = np.unravel_index(M.argmax(), M.shape)
    return (np.tril(M,-1).sum(), np.trace(M), np.triu(M,1).sum(), f"{i}\u2013{j}", lh, la, M)

# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.set_page_config(page_title="2026 World Cup Predictor", page_icon="\u26bd", layout="wide")
st.title("\u26bd 2026 World Cup Predictor")
st.caption(f"Elo-seeded Poisson goals model + Monte Carlo simulator · "
           f"trained on results through {manifest['model_data_through']} · "
           f"locked {manifest['generated_at_utc']} · {manifest['n_simulations']:,} simulations")

tab_odds, tab_games, tab_predict, tab_about = st.tabs(
    ["\U0001F3C6 Title Odds", "\U0001F4C5 Match Predictions", "\U0001F52E Predict a Match", "\u2139\ufe0f How it works"])

# ---- Tab 1: title odds ----
with tab_odds:
    st.subheader("Who wins the 2026 World Cup?")
    top = odds.head(15)
    chart = (alt.Chart(top).mark_bar(color="#1c8a7a")
             .encode(x=alt.X("champion:Q", title="Probability of winning", axis=alt.Axis(format="%")),
                     y=alt.Y("team:N", sort="-x", title=None),
                     tooltip=["team", alt.Tooltip("champion:Q", format=".1%")])
             .properties(height=460))
    st.altair_chart(chart, width='stretch')
    st.markdown("##### Advancement probabilities (all teams)")
    show = odds.copy()
    pct = {c: st.column_config.ProgressColumn(c.replace("reach_", "").replace("_", " ").title(),
            format="%.1f%%", min_value=0, max_value=1)
           for c in ["reach_R32","reach_R16","reach_QF","reach_SF","reach_final","champion"]}
    st.dataframe(show, hide_index=True, width='stretch', column_config=pct)

# ---- Tab 2: every group-stage match ----
with tab_games:
    st.subheader("Group-stage predictions (locked before kickoff)")
    teamf = st.selectbox("Filter by team", ["All teams"] + TEAMS)
    g = games if teamf == "All teams" else games[(games.home == teamf) | (games.away == teamf)]
    disp = g.assign(match=g.home + "  v  " + g.away)[
        ["date","match","favorite","likely_score","p_home","p_draw","p_away","xg_home","xg_away"]]
    st.dataframe(disp, hide_index=True, width='stretch', column_config={
        "p_home": st.column_config.NumberColumn("P(home)", format="%.0f%%"),
        "p_draw": st.column_config.NumberColumn("P(draw)", format="%.0f%%"),
        "p_away": st.column_config.NumberColumn("P(away)", format="%.0f%%")})
    st.caption("p_home/draw/away shown as fractions in the data; favourite is the most likely single outcome.")

# ---- Tab 3: interactive match predictor ----
with tab_predict:
    st.subheader("Predict any match-up")
    c1, c2, c3 = st.columns([3, 3, 2])
    home = c1.selectbox("Home / Team A", TEAMS, index=TEAMS.index("Argentina"))
    away = c2.selectbox("Away / Team B", TEAMS, index=TEAMS.index("France"))
    neutral = c3.toggle("Neutral venue", value=True)
    if home == away:
        st.warning("Pick two different teams.")
    else:
        pH, pD, pA, ml, lh, la, M = predict(home, away, neutral)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{home} win", f"{pH:.0%}")
        m2.metric("Draw", f"{pD:.0%}")
        m3.metric(f"{away} win", f"{pA:.0%}")
        m4.metric("Likely score", ml, help="Most probable exact scoreline")
        st.caption(f"Expected goals — {home}: {lh:.2f} · {away}: {la:.2f}")

        # full scoreline distribution as a heatmap
        mm = M[:6, :6]
        hm = pd.DataFrame([(i, j, mm[i, j]) for i in range(6) for j in range(6)],
                          columns=["home_goals", "away_goals", "prob"])
        heat = (alt.Chart(hm).mark_rect()
                .encode(x=alt.X("away_goals:O", title=f"{away} goals"),
                        y=alt.Y("home_goals:O", title=f"{home} goals", sort="descending"),
                        color=alt.Color("prob:Q", scale=alt.Scale(scheme="teals"), legend=None),
                        tooltip=[alt.Tooltip("prob:Q", format=".1%")])
                .properties(height=320, title="Probability of each exact scoreline"))
        st.altair_chart(heat, width='stretch')

# ---- Tab 4: methodology ----
with tab_about:
    st.markdown((DATA / "LOCK.md").read_text())

st.divider()
st.caption("Predictions are committed to git before each round — no hindsight. "
           "Model rates teams from results only; it cannot see injuries or squad news.")
