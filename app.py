"""
app.py — 2026 World Cup forecasting dashboard.

Tabs:
  Title Odds        — Monte Carlo champion + advancement probabilities
  Match Predictions — livescore-style scoreboard (default) or a sortable table
  Predict a Match   — interactive predictor with scoreline heatmap

Run locally:  streamlit run app.py
Deploy:       push to GitHub -> share.streamlit.io -> point at this file.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from scipy.stats import poisson

DATA = Path(__file__).parent / "outputs"

# 2026 tri-host palette (USA / Canada / Mexico): red, blue, green
RED, BLUE, GREEN = "#E4002B", "#0A3161", "#006341"
INK, MUTED, LINE, CARD = "#0d1b2a", "#5b6b7d", "#e6e9ee", "#ffffff"

@st.cache_data
def load_model():
    S = pd.read_csv(DATA / "team_strengths.csv", index_col=0)
    meta = json.load(open(DATA / "_glm_meta.json"))
    return S["attack"].to_dict(), S["defend"].to_dict(), meta

@st.cache_data
def load_predictions():
    odds = pd.read_csv(DATA / "locked_title_odds.csv")
    games = pd.read_csv(DATA / "locked_group_predictions.csv", parse_dates=["date"])
    manifest = json.load(open(DATA / "lock_manifest.json"))
    return odds, games, manifest

att, dfn, META = load_model()
INT, HOME, RHO = META["intercept"], META["home"], META["rho"]
odds, games, manifest = load_predictions()
TEAMS = sorted(att.keys())

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

st.set_page_config(page_title="2026 World Cup Predictor", page_icon="\u26bd", layout="wide")
st.markdown(f"""
<style>
.block-container {{padding-top: 2rem;}}
.match-card {{
  background:{CARD}; border:1px solid {LINE}; border-left:6px solid {BLUE};
  border-radius:12px; padding:14px 18px; margin-bottom:10px;
}}
.mc-top {{display:flex; align-items:center; justify-content:space-between;}}
.mc-team {{font-size:1.05rem; font-weight:600; color:{INK}; flex:1;}}
.mc-team.away {{text-align:right;}}
.mc-score {{font-size:1.5rem; font-weight:800; color:{INK}; padding:0 16px; letter-spacing:1px;}}
.mc-bar {{display:flex; height:8px; border-radius:5px; overflow:hidden; margin:10px 0 6px;}}
.mc-meta {{display:flex; justify-content:space-between; font-size:0.8rem; color:{MUTED};}}
.mc-fav {{font-size:0.78rem; color:{RED}; font-weight:600;}}
</style>
""", unsafe_allow_html=True)

st.title("\u26bd 2026 World Cup Predictor")
st.caption(f"Elo-seeded Poisson goals model + Monte Carlo simulator \u00b7 "
           f"trained through {manifest['model_data_through']} \u00b7 "
           f"locked {manifest['generated_at_utc']} \u00b7 {manifest['n_simulations']:,} simulations")

tab_odds, tab_board, tab_predict = st.tabs(
    ["\U0001F3C6 Title Odds", "\U0001F4C5 Match Predictions", "\U0001F52E Predict a Match"])

with tab_odds:
    st.subheader("Who wins the 2026 World Cup?")
    top = odds.head(15)
    chart = (alt.Chart(top).mark_bar(color=BLUE)
             .encode(x=alt.X("champion:Q", title="Probability of winning", axis=alt.Axis(format="%")),
                     y=alt.Y("team:N", sort="-x", title=None),
                     tooltip=["team", alt.Tooltip("champion:Q", format=".1%")])
             .properties(height=460))
    st.altair_chart(chart, width='stretch')
    st.markdown("##### Advancement probabilities (all teams)")
    pct = {c: st.column_config.ProgressColumn(c.replace("reach_","").replace("_"," ").title(),
            format="%.1f%%", min_value=0, max_value=1)
           for c in ["reach_R32","reach_R16","reach_QF","reach_SF","reach_final","champion"]}
    st.dataframe(odds, hide_index=True, width='stretch', column_config=pct)

def match_card(r):
    tot = r.p_home + r.p_draw + r.p_away
    wh, wd, wa = (r.p_home/tot*100, r.p_draw/tot*100, r.p_away/tot*100)
    return f"""
    <div class="match-card">
      <div class="mc-top">
        <div class="mc-team">{r.home}</div>
        <div class="mc-score">{r.likely_score}</div>
        <div class="mc-team away">{r.away}</div>
      </div>
      <div class="mc-bar">
        <div style="width:{wh}%; background:{GREEN};"></div>
        <div style="width:{wd}%; background:{MUTED};"></div>
        <div style="width:{wa}%; background:{RED};"></div>
      </div>
      <div class="mc-meta">
        <span>Win {wh:.0f}%</span>
        <span>Draw {wd:.0f}%</span>
        <span>Win {wa:.0f}%</span>
      </div>
      <div class="mc-meta" style="margin-top:4px;">
        <span>xG {r.xg_home:.2f}</span>
        <span class="mc-fav">Favourite: {r.favorite}</span>
        <span>xG {r.xg_away:.2f}</span>
      </div>
    </div>"""

with tab_board:
    view = st.radio("View", ["Scoreboard", "Table"], horizontal=True, label_visibility="collapsed")
    st.caption("Predicted results for every group-stage match \u2014 locked before kickoff. "
               "Green = home win likelihood, grey = draw, red = away win.")
    if view == "Scoreboard":
        for day, block in games.groupby(games["date"].dt.date):
            label = pd.Timestamp(day).strftime("%A, %d %B %Y")
            with st.expander(f"{label}   \u2014   {len(block)} matches", expanded=True):
                for r in block.itertuples():
                    st.markdown(match_card(r), unsafe_allow_html=True)
    else:
        t = games.assign(date=games["date"].dt.date,
                         match=games.home + " v " + games.away)[
            ["date","match","favorite","likely_score","p_home","p_draw","p_away","xg_home","xg_away"]]
        st.dataframe(t, hide_index=True, width='stretch', column_config={
            "p_home": st.column_config.NumberColumn("P(home)", format="%.0f%%"),
            "p_draw": st.column_config.NumberColumn("P(draw)", format="%.0f%%"),
            "p_away": st.column_config.NumberColumn("P(away)", format="%.0f%%")})

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
        m4.metric("Likely score", ml)
        st.caption(f"Expected goals \u2014 {home}: {lh:.2f} \u00b7 {away}: {la:.2f}")
        mm = M[:6, :6]
        hm = pd.DataFrame([(i, j, mm[i, j]) for i in range(6) for j in range(6)],
                          columns=["home_goals", "away_goals", "prob"])
        heat = (alt.Chart(hm).mark_rect()
                .encode(x=alt.X("away_goals:O", title=f"{away} goals"),
                        y=alt.Y("home_goals:O", title=f"{home} goals", sort="descending"),
                        color=alt.Color("prob:Q", scale=alt.Scale(scheme="blues"), legend=None),
                        tooltip=[alt.Tooltip("prob:Q", format=".1%")])
                .properties(height=320, title="Probability of each exact scoreline"))
        st.altair_chart(heat, width='stretch')

st.divider()
st.caption("Predictions committed to git before each round \u2014 no hindsight. "
           "Model rates teams from results only; it cannot see injuries or squad news.")
