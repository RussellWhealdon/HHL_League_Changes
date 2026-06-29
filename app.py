import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(
    page_title="HHL Rule Change Explorer",
    page_icon="🏈",
    layout="wide",
)

STARTER_SLOTS  = {"QB", "RB", "WR", "TE", "FLEX", "K", "D/ST"}
PLAYOFF_WEEKS  = {15, 16, 17}

NUMERIC_PS = [
    "fantasy_points_ppr", "rb_first_downs", "rushing_yards",
    "fg_made_0_19", "fg_made_20_29", "fg_made_30_39",
    "fg_made_40_49", "fg_made_50_59", "fg_made_60_", "pat_made",
]
NUMERIC_HHL = [
    "ps_fantasy_points_ppr", "ps_rb_first_downs", "ps_rushing_yards",
    "ps_fg_made_0_19", "ps_fg_made_20_29", "ps_fg_made_30_39",
    "ps_fg_made_40_49", "ps_fg_made_50_59", "ps_fg_made_60_",
    "ps_pat_made", "dst_fantasy_pts",
]


# ── Data ─────────────────────────────────────────────────────────────────────

@st.cache_data
def load_player_data():
    ps = pd.read_csv("data/player_stats.csv", low_memory=False)
    ps = ps[
        (ps["season_type"] == "REG") &
        (ps["position"].isin({"QB", "RB", "WR", "TE", "K"}))
    ].copy()
    for col in NUMERIC_PS:
        ps[col] = pd.to_numeric(ps[col], errors="coerce").fillna(0)

    dst_raw = pd.read_csv("data/fantasy_points_per_game.csv", low_memory=False)
    dst = dst_raw[dst_raw["position"] == "DEF"].copy()
    dst["fantasy_pts"] = pd.to_numeric(dst["fantasy_pts"], errors="coerce").fillna(0)
    dst["player_display_name"] = dst["player_display_name"].str.replace(" Defense", "", regex=False)
    return ps, dst


@st.cache_data
def load_hhl_data():
    hhl = pd.read_csv("data/hhl_rosters_with_stats.csv")
    for col in NUMERIC_HHL:
        if col in hhl.columns:
            hhl[col] = pd.to_numeric(hhl[col], errors="coerce").fillna(0)
    hhl["Week"] = pd.to_numeric(hhl["Week"], errors="coerce")

    teams = pd.read_csv("data/fantasy_teams.csv")
    name_to_id = dict(zip(teams["fantasy_team_name"], teams["fantasy_team_id"]))

    # Actual ESPN scores — strip records like "(8-6-0)" from team name column
    raw = pd.read_csv("data/hhl_schedule_scores.csv")
    import re
    def strip_record(name):
        return re.sub(r"\([\d-]+\)$", "", str(name)).strip()

    raw["away_team_id"] = raw["AWAY TEAM"].apply(strip_record).map(name_to_id)
    raw["home_team_id"] = raw["HOME TEAM"].apply(strip_record).map(name_to_id)
    actual_scores = raw[["Week", "away_team_id", "home_team_id", "Away Score", "Home Score"]].rename(
        columns={"Away Score": "away_actual", "Home Score": "home_actual"}
    )

    return hhl, actual_scores, teams


# ── Scoring: Player page ──────────────────────────────────────────────────────

def compute_skill_score(ps: pd.DataFrame, rules: dict) -> pd.DataFrame:
    ps = ps.copy()
    is_kicker = ps["position"] == "K"
    is_skill  = ~is_kicker

    base = pd.Series(0.0, index=ps.index)
    ki   = ps.index[is_kicker]
    kdf  = ps.loc[ki]
    base[ki] = (
        kdf["fg_made_0_19"] * 3 + kdf["fg_made_20_29"] * 3 + kdf["fg_made_30_39"] * 3
        + kdf["fg_made_40_49"] * 4 + kdf["fg_made_50_59"] * 5 + kdf["fg_made_60_"] * 5
        + kdf["pat_made"] * 1
    )
    base[is_skill] = ps.loc[is_skill, "fantasy_points_ppr"]

    adj = base.copy()
    if rules.get("halve_kicker"):
        adj[is_kicker] *= 0.5
    fd = rules.get("rush_fd", 0)
    if fd:
        adj[is_skill] += ps.loc[is_skill, "rb_first_downs"] * fd
    b100 = rules.get("rush_100_bonus", 0)
    if b100:
        adj[is_skill & (ps["rushing_yards"] >= 100)] += b100

    ps["base_score"] = base
    ps["adj_score"]  = adj
    return ps


def compute_dst_score(dst: pd.DataFrame, rules: dict) -> pd.DataFrame:
    dst = dst.copy()
    dst["base_score"] = dst["fantasy_pts"]
    dst["adj_score"]  = 0.0 if rules.get("remove_dst") else dst["fantasy_pts"]
    return dst


def build_leaderboard(ps, dst, rules):
    ps_s  = compute_skill_score(ps, rules)
    dst_s = compute_dst_score(dst, rules)

    skill = (
        ps_s.groupby(["player_display_name", "position"])
        .agg(Base=("base_score", "sum"), Adjusted=("adj_score", "sum"), Weeks=("adj_score", "count"))
        .reset_index()
        .rename(columns={"player_display_name": "Player", "position": "Position"})
    )
    d = (
        dst_s.groupby("player_display_name")
        .agg(Base=("base_score", "sum"), Adjusted=("adj_score", "sum"), Weeks=("adj_score", "count"))
        .reset_index()
        .rename(columns={"player_display_name": "Player"})
    )
    d["Position"] = "DST"

    out = pd.concat([skill, d], ignore_index=True)
    out["Delta"] = out["Adjusted"] - out["Base"]
    out = out.sort_values("Adjusted", ascending=False).reset_index(drop=True)
    out.index += 1
    return out


# ── Scoring: Team page ────────────────────────────────────────────────────────

def compute_hhl_scores(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    df = df.copy()
    is_dst    = df["SLOT"] == "D/ST"
    is_kicker = df["ps_position"] == "K"
    is_skill  = ~is_dst & ~is_kicker

    base = pd.Series(0.0, index=df.index)
    base[is_dst] = df.loc[is_dst, "dst_fantasy_pts"]
    ki  = df.index[is_kicker]
    kdf = df.loc[ki]
    base[ki] = (
        kdf["ps_fg_made_0_19"] * 3 + kdf["ps_fg_made_20_29"] * 3 + kdf["ps_fg_made_30_39"] * 3
        + kdf["ps_fg_made_40_49"] * 4 + kdf["ps_fg_made_50_59"] * 5 + kdf["ps_fg_made_60_"] * 5
        + kdf["ps_pat_made"] * 1
    )
    base[is_skill] = df.loc[is_skill, "ps_fantasy_points_ppr"]

    adj = base.copy()
    if rules.get("halve_kicker"):
        adj[is_kicker] *= 0.5
    if rules.get("remove_dst"):
        adj[is_dst] = 0
    fd = rules.get("rush_fd", 0)
    if fd:
        adj[is_skill] += df.loc[is_skill, "ps_rb_first_downs"] * fd
    b100 = rules.get("rush_100_bonus", 0)
    if b100:
        adj[is_skill & (df["ps_rushing_yards"] >= 100)] += b100

    df["base_score"] = base
    df["adj_score"]  = adj
    return df


def get_team_weekly(hhl, rules, include_playoffs):
    scored   = compute_hhl_scores(hhl, rules)
    starters = scored[scored["SLOT"].isin(STARTER_SLOTS)].copy()
    if not include_playoffs:
        starters = starters[~starters["Week"].isin(PLAYOFF_WEEKS)]
    return (
        starters.groupby(["Team", "Week"])
        .agg(base_pts=("base_score", "sum"), adj_pts=("adj_score", "sum"))
        .reset_index()
    )


def get_game_scores(actual_scores, team_week, include_playoffs):
    """Actual ESPN scores as base; add computed rule-change deltas for adjusted scores."""
    sched = actual_scores.copy()
    if not include_playoffs:
        sched = sched[~sched["Week"].isin(PLAYOFF_WEEKS)]

    # Merge computed base + adj per team-week to derive the delta
    sched = sched.merge(
        team_week.rename(columns={"Team": "home_team_id", "base_pts": "h_comp_base", "adj_pts": "h_comp_adj"}),
        on=["home_team_id", "Week"], how="left",
    )
    sched = sched.merge(
        team_week.rename(columns={"Team": "away_team_id", "base_pts": "a_comp_base", "adj_pts": "a_comp_adj"}),
        on=["away_team_id", "Week"], how="left",
    )
    for c in ["h_comp_base", "h_comp_adj", "a_comp_base", "a_comp_adj"]:
        sched[c] = sched[c].fillna(0)

    # Adjusted = actual ESPN score + delta from rule changes
    sched["home_adj"] = sched["home_actual"] + (sched["h_comp_adj"] - sched["h_comp_base"])
    sched["away_adj"] = sched["away_actual"] + (sched["a_comp_adj"] - sched["a_comp_base"])

    sched["base_home_win"] = sched["home_actual"] > sched["away_actual"]
    sched["adj_home_win"]  = sched["home_adj"]    > sched["away_adj"]
    sched["flipped"]       = sched["base_home_win"] != sched["adj_home_win"]
    return sched


def get_standings(game_scores, teams, score: str):
    # score = "actual" (base) or "adj"
    hw  = "base_home_win" if score == "actual" else "adj_home_win"
    hp  = "home_actual"   if score == "actual" else "home_adj"
    ap  = "away_actual"   if score == "actual" else "away_adj"

    home = game_scores.groupby("home_team_id").agg(
        hpf=(hp, "sum"), hpa=(ap, "sum"), hw=(hw, "sum"), hg=(hp, "count")
    ).reset_index().rename(columns={"home_team_id": "tid"})
    home["hl"] = home["hg"] - home["hw"]

    aw_s = (~game_scores[hw]).groupby(game_scores["away_team_id"]).sum().reset_index()
    aw_s.columns = ["tid", "aw"]
    away = game_scores.groupby("away_team_id").agg(
        apf=(ap, "sum"), apa=(hp, "sum"), ag=(ap, "count")
    ).reset_index().rename(columns={"away_team_id": "tid"})
    away = away.merge(aw_s, on="tid")
    away["al"] = away["ag"] - away["aw"]

    s = home.merge(away, on="tid")
    s["PF"]     = (s["hpf"] + s["apf"]).round(2)
    s["PA"]     = (s["hpa"] + s["apa"]).round(2)
    s["W"]      = (s["hw"] + s["aw"]).astype(int)
    s["L"]      = (s["hl"] + s["al"]).astype(int)
    s["Record"] = s["W"].astype(str) + "-" + s["L"].astype(str)

    s = s.merge(teams, left_on="tid", right_on="fantasy_team_id", how="left")
    s = s.sort_values(["W", "PF"], ascending=False).reset_index(drop=True)
    s.index += 1
    return s[["fantasy_team_name", "owner_name", "Record", "W", "L", "PF", "PA"]].rename(
        columns={"fantasy_team_name": "Team", "owner_name": "Owner"}
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏈 HHL Explorer")
    st.divider()
    page = st.radio("", ["Player Rankings", "Team Standings"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Rule Changes**")
    halve_kicker   = st.checkbox("⚡ Halve kicker points")
    remove_dst     = st.checkbox("🚫 Remove defenses (D/ST)")
    rush_fd        = st.slider("🏃 Rushing 1st down bonus", 0.0, 1.0, 0.0, 0.25, format="+%.2f pts")
    rush_100_bonus = st.slider("💯 100+ rush yard bonus", 0, 10, 0, 1, format="+%d pts")

rules = {
    "halve_kicker":   halve_kicker,
    "remove_dst":     remove_dst,
    "rush_fd":        rush_fd,
    "rush_100_bonus": rush_100_bonus,
}
any_rule_active = any([halve_kicker, remove_dst, rush_fd > 0, rush_100_bonus > 0])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Player Rankings
# ══════════════════════════════════════════════════════════════════════════════

if page == "Player Rankings":
    st.title("Player Rankings")
    st.caption("2025 NFL Regular Season · All players · Adjust rule changes in the sidebar.")

    ps, dst = load_player_data()
    leaderboard = build_leaderboard(ps, dst, rules)

    score_col = "Adjusted" if any_rule_active else "Base"

    positions  = sorted(leaderboard["Position"].unique().tolist())
    fc1, fc2   = st.columns([2, 3])
    pos_filter = fc1.multiselect("Position", positions, placeholder="All positions")
    n_players  = fc2.slider("Show top N players", 10, 200, 30, 10)

    filtered = leaderboard.copy()
    if pos_filter:
        filtered = filtered[filtered["Position"].isin(pos_filter)]
    filtered = filtered.head(n_players)

    # Rankings table
    with st.expander("Player Rankings", expanded=True):
        if not any_rule_active:
            display = filtered[["Player", "Position", "Weeks", "Base"]].rename(columns={"Base": "Season Pts"})
            display["Season Pts"] = display["Season Pts"].round(1)
            st.dataframe(display, use_container_width=True, height=520,
                         column_config={"Season Pts": st.column_config.NumberColumn(format="%.1f")})
        else:
            display = filtered[["Player", "Position", "Weeks", "Base", "Adjusted", "Delta"]].rename(
                columns={"Base": "Base Pts", "Adjusted": "Adj Pts", "Delta": "Δ Pts"})
            for c in ["Base Pts", "Adj Pts", "Δ Pts"]:
                display[c] = display[c].round(1)
            st.dataframe(display, use_container_width=True, height=520,
                         column_config={k: st.column_config.NumberColumn(format="%.1f")
                                        for k in ["Base Pts", "Adj Pts", "Δ Pts"]})

            m1, m2, m3 = st.columns(3)
            m1.metric("Base Total", f"{filtered['Base'].sum():,.1f} pts")
            m2.metric("Adjusted Total", f"{filtered['Adjusted'].sum():,.1f} pts")
            m3.metric("Δ Total", f"{filtered['Adjusted'].sum() - filtered['Base'].sum():+,.1f} pts")

    # PPG bar chart
    st.markdown("### Avg PPG — Top 30% of Players per Position")
    st.caption("Average points per game across the top 30% of players at each position by season total.")

    ppg_df = leaderboard.copy()
    ppg_df["PPG"] = (ppg_df[score_col] / ppg_df["Weeks"].clip(lower=1)).round(2)
    top30 = pd.concat([
        g.nlargest(max(1, int(len(g) * 0.30)), score_col)
        for _, g in ppg_df.groupby("Position")
            ]).reset_index(drop=True)

    avg_ppg = (
        top30.groupby("Position")
        .agg(avg_ppg_val=("PPG", "mean"), n=("PPG", "count"))
        .round({"avg_ppg_val": 2})
        .reset_index()
        .rename(columns={"avg_ppg_val": "Avg PPG"})
        .sort_values("Avg PPG", ascending=False)
    )
    avg_ppg["label"] = avg_ppg["Avg PPG"].round(1).astype(str) + " — " + avg_ppg["n"].astype(str) + " players"
    avg_ppg.loc[avg_ppg["Position"] == "DST", "label"] = (
        avg_ppg.loc[avg_ppg["Position"] == "DST", "Avg PPG"].round(1).astype(str)
        + " — " + avg_ppg.loc[avg_ppg["Position"] == "DST", "n"].astype(str) + " teams"
    )

    bars = (
        alt.Chart(avg_ppg)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Position:N", sort="-y", axis=alt.Axis(labelFontSize=13)),
            y=alt.Y("Avg PPG:Q", title="Avg PPG (Top 30%)", axis=alt.Axis(grid=True)),
            color=alt.Color("Position:N", legend=None),
            tooltip=["Position", alt.Tooltip("Avg PPG:Q", format=".2f"), alt.Tooltip("label:N", title="Sample")],
        )
        .properties(height=380)
    )
    text = (
        alt.Chart(avg_ppg)
        .mark_text(dy=-8, fontSize=12, color="white")
        .encode(
            x=alt.X("Position:N", sort="-y"),
            y=alt.Y("Avg PPG:Q"),
            text=alt.Text("label:N"),
        )
    )
    st.altair_chart(bars + text, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Team Standings
# ══════════════════════════════════════════════════════════════════════════════

else:
    st.title("Team Standings")
    st.caption("HHL 2025 · Scores calculated from starter lineups each week.")

    hhl, actual_scores, teams = load_hhl_data()
    tid_to_name = dict(zip(teams["fantasy_team_id"], teams["fantasy_team_name"]))

    include_playoffs = st.toggle("Include playoffs (Weeks 15–17)", value=False)

    team_week = get_team_weekly(hhl, rules, include_playoffs)
    sched_res  = get_game_scores(actual_scores, team_week, include_playoffs)

    base_std = get_standings(sched_res, teams, "actual")
    adj_std  = get_standings(sched_res, teams, "adj")

    # ── Standings expander ────────────────────────────────────────────────────
    with st.expander("Team Standings", expanded=True):
        if not any_rule_active:
            st.dataframe(base_std, use_container_width=True,
                         column_config={
                             "PF": st.column_config.NumberColumn(format="%.1f"),
                             "PA": st.column_config.NumberColumn(format="%.1f"),
                         })
        else:
            # Merge to flag record changes
            merged = base_std.rename(columns={"Record": "Base Record", "W": "Base W", "L": "Base L",
                                               "PF": "Base PF", "PA": "Base PA"}) \
                             .merge(adj_std.rename(columns={"Record": "Adj Record", "W": "Adj W", "L": "Adj L",
                                                            "PF": "Adj PF", "PA": "Adj PA"}),
                                    on=["Team", "Owner"])
            # Count flipped games per team (home + away appearances)
            flipped_home = sched_res[sched_res["flipped"]].groupby("home_team_id")["flipped"].sum().reset_index()
            flipped_home.columns = ["fantasy_team_id", "flips"]
            flipped_away = sched_res[sched_res["flipped"]].groupby("away_team_id")["flipped"].sum().reset_index()
            flipped_away.columns = ["fantasy_team_id", "flips"]
            flips_by_team = pd.concat([flipped_home, flipped_away]).groupby("fantasy_team_id")["flips"].sum().reset_index()
            flips_by_team = flips_by_team.merge(teams[["fantasy_team_id", "fantasy_team_name"]], on="fantasy_team_id")
            flips_by_team = flips_by_team.rename(columns={"fantasy_team_name": "Team"})

            merged = base_std.rename(columns={"Record": "Base Record", "W": "Base W", "L": "Base L",
                                               "PF": "Base PF", "PA": "Base PA"}) \
                             .merge(adj_std.rename(columns={"Record": "Adj Record", "W": "Adj W", "L": "Adj L",
                                                            "PF": "Adj PF", "PA": "Adj PA"}),
                                    on=["Team", "Owner"])
            merged = merged.merge(flips_by_team[["Team", "flips"]], on="Team", how="left")
            merged["Games Flipped"] = merged["flips"].fillna(0).astype(int)
            merged = merged.sort_values("Adj W", ascending=False).reset_index(drop=True)
            merged.index += 1

            st.dataframe(
                merged[["Team", "Owner", "Base Record", "Adj Record", "Games Flipped",
                        "Base PF", "Adj PF", "Base PA", "Adj PA"]],
                use_container_width=True,
                column_config={
                    "Base PF": st.column_config.NumberColumn(format="%.2f"),
                    "Adj PF":  st.column_config.NumberColumn(format="%.2f"),
                    "Base PA": st.column_config.NumberColumn(format="%.2f"),
                    "Adj PA":  st.column_config.NumberColumn(format="%.2f"),
                },
            )
            flipped_games = int(sched_res["flipped"].sum())
            if flipped_games:
                st.caption(f"⚠️ {flipped_games} game result{'s' if flipped_games > 1 else ''} flip with these rule changes.")

    # ── Full Schedule ─────────────────────────────────────────────────────────
    st.markdown("### Full Schedule")

    sched_display = sched_res.copy()
    sched_display["Home Team"] = sched_display["home_team_id"].map(tid_to_name)
    sched_display["Away Team"] = sched_display["away_team_id"].map(tid_to_name)

    if not any_rule_active:
        sched_display["Home Score"] = sched_display["home_actual"].round(2)
        sched_display["Away Score"] = sched_display["away_actual"].round(2)
        sched_display["Winner"] = np.where(
            sched_display["base_home_win"], sched_display["Home Team"], sched_display["Away Team"]
        )
        show_cols = ["Week", "Away Team", "Away Score", "Home Score", "Home Team", "Winner"]
    else:
        sched_display["Away Base"] = sched_display["away_actual"].round(2)
        sched_display["Home Base"] = sched_display["home_actual"].round(2)
        sched_display["Away Adj"]  = sched_display["away_adj"].round(1)
        sched_display["Home Adj"]  = sched_display["home_adj"].round(1)
        sched_display["Base Winner"] = np.where(
            sched_display["base_home_win"], sched_display["Home Team"], sched_display["Away Team"]
        )
        sched_display["Adj Winner"] = np.where(
            sched_display["adj_home_win"], sched_display["Home Team"], sched_display["Away Team"]
        )
        sched_display["🔄"] = sched_display["flipped"].map({True: "🔄 Flipped", False: ""})
        show_cols = ["Week", "Away Team", "Away Base", "Away Adj", "Home Adj", "Home Base", "Home Team", "Base Winner", "Adj Winner", "🔄"]

    st.dataframe(
        sched_display[show_cols].sort_values("Week").reset_index(drop=True),
        use_container_width=True,
        height=500,
        column_config={c: st.column_config.NumberColumn(format="%.1f")
                       for c in show_cols if "Score" in c or "Base" in c or "Adj" in c
                       if c not in ["Away Team", "Home Team", "Base Winner", "Adj Winner"]},
    )

    # ── Game Detail ───────────────────────────────────────────────────────────
    st.markdown("### Game Detail")
    scored_hhl = compute_hhl_scores(hhl, rules)
    if not include_playoffs:
        scored_hhl = scored_hhl[~scored_hhl["Week"].isin(PLAYOFF_WEEKS)]

    game_options = [
        f"Wk {row['Week']} · {tid_to_name.get(row['away_team_id'], row['away_team_id'])} @ {tid_to_name.get(row['home_team_id'], row['home_team_id'])}"
        for _, row in sched_res.sort_values(["Week"]).iterrows()
    ]
    selected = st.selectbox("Select a game", game_options)

    if selected:
        idx   = game_options.index(selected)
        game  = sched_res.sort_values("Week").iloc[idx]
        week  = int(game["Week"])
        home_id, away_id = game["home_team_id"], game["away_team_id"]

        score_key = "adj_score" if any_rule_active else "base_score"

        def get_lineup(team_id):
            rows = scored_hhl[
                (scored_hhl["Team"] == team_id) &
                (scored_hhl["Week"] == week) &
                (scored_hhl["SLOT"].isin(STARTER_SLOTS))
            ][["SLOT", "Player", "ps_position", score_key]].copy()
            rows["Pos"] = np.where(rows["SLOT"] == "D/ST", "DST", rows["ps_position"].fillna("—"))
            rows = rows.rename(columns={score_key: "Pts", "SLOT": "Slot"})
            rows["Pts"] = rows["Pts"].round(1)
            return rows[["Slot", "Player", "Pos", "Pts"]].sort_values("Pts", ascending=False).reset_index(drop=True)

        home_lineup = get_lineup(home_id)
        away_lineup = get_lineup(away_id)

        home_total = home_lineup["Pts"].sum()
        away_total = away_lineup["Pts"].sum()

        col_a, col_h = st.columns(2)
        with col_a:
            st.markdown(f"**{tid_to_name.get(away_id, away_id)}** · {away_total:.1f} pts")
            st.dataframe(away_lineup, use_container_width=True, hide_index=True,
                         column_config={"Pts": st.column_config.NumberColumn(format="%.1f")})
        with col_h:
            st.markdown(f"**{tid_to_name.get(home_id, home_id)}** · {home_total:.1f} pts")
            st.dataframe(home_lineup, use_container_width=True, hide_index=True,
                         column_config={"Pts": st.column_config.NumberColumn(format="%.1f")})
