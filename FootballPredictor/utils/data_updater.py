import os
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from utils.name_mapping import normalize_team_name

load_dotenv()

HISTORICAL_CSV = "data/historical_matches.csv"
LAST_UPDATE_FILE = "data/last_update.txt"
APIFOOTBALL_BASE = "https://apiv3.apifootball.com/"
PL_LEAGUE_ID = 152

CSV_COLUMNS = [
    "date", "homeTeam", "awayTeam", "homeGoals", "awayGoals",
    "homePossession", "awayPossession", "homeShots", "awayShots",
    "homeShotsOnTarget", "awayShotsOnTarget",
]


def read_last_update():
    if not os.path.exists(LAST_UPDATE_FILE):
        return datetime(2000, 1, 1, tzinfo=timezone.utc)
    with open(LAST_UPDATE_FILE, "r") as f:
        ts = f.read().strip()
    try:
        return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    except Exception:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)


def save_last_update():
    with open(LAST_UPDATE_FILE, "w") as f:
        f.write(datetime.now(timezone.utc).isoformat())


def _get_stat(statistics, stat_type, side):
    """Pull a numeric value from the apifootball statistics list."""
    for s in statistics:
        if s.get("type") == stat_type:
            val = s.get(side, "")
            if val is None:
                return ""
            if isinstance(val, str) and val.endswith("%"):
                val = val.rstrip("%")
            try:
                return float(val)
            except (ValueError, TypeError):
                return ""
    return ""


def _fetch_events(api_key, date_from, date_to):
    """Return all finished PL matches from apifootball.com for a date range."""
    resp = requests.get(APIFOOTBALL_BASE, params={
        "action": "get_events",
        "league_id": PL_LEAGUE_ID,
        "from": date_from.strftime("%Y-%m-%d"),
        "to": date_to.strftime("%Y-%m-%d"),
        "APIkey": api_key,
    })
    if resp.status_code != 200:
        raise Exception(f"apifootball error {resp.status_code}: {resp.text}")
    data = resp.json()
    if isinstance(data, dict):
        # apifootball returns {"error": 404, "message": "No event found..."} for empty ranges
        if data.get("error") == 404:
            return []
        raise Exception(f"apifootball API error: {data}")
    return [m for m in data if m.get("match_status") == "Finished"]


def _convert_match(match):
    """Convert an apifootball event dict into a CSV row dict."""
    dt = datetime.strptime(
        f"{match['match_date']} {match.get('match_time', '00:00')}",
        "%Y-%m-%d %H:%M"
    )
    stats = match.get("statistics", [])

    home_poss = _get_stat(stats, "Ball Possession", "home")
    away_poss = _get_stat(stats, "Ball Possession", "away")
    # apifootball uses 0%/0% as a sentinel when possession wasn't tracked
    if home_poss == 0.0 and away_poss == 0.0:
        home_poss = ""
        away_poss = ""

    return {
        "date": dt.strftime("%d/%m/%Y %H:%M"),
        "homeTeam": normalize_team_name(match["match_hometeam_name"]),
        "awayTeam": normalize_team_name(match["match_awayteam_name"]),
        "homeGoals": match.get("match_hometeam_score", ""),
        "awayGoals": match.get("match_awayteam_score", ""),
        "homePossession":     home_poss,
        "awayPossession":     away_poss,
        "homeShots":          _get_stat(stats, "Shots Total", "home"),
        "awayShots":          _get_stat(stats, "Shots Total", "away"),
        "homeShotsOnTarget":  _get_stat(stats, "Shots On Goal", "home"),
        "awayShotsOnTarget":  _get_stat(stats, "Shots On Goal", "away"),
    }


def _backfill_missing_stats(df, api_key):
    """Fill stats columns for any rows already in the CSV that are still empty."""
    missing_mask = df["homePossession"].isna() | (df["homePossession"] == "")
    if not missing_mask.any():
        return df, 0

    missing_df = df[missing_mask]
    dates = pd.to_datetime(missing_df["date"], format="%d/%m/%Y %H:%M")
    date_from = dates.min().normalize()
    date_to   = dates.max().normalize()

    print(f"  Backfilling stats for {len(missing_df)} row(s) "
          f"({date_from.date()} to {date_to.date()})...")

    try:
        matches = _fetch_events(api_key, date_from, date_to)
    except Exception as e:
        print(f"  Could not fetch backfill data: {e}")
        return df, 0

    # Build lookup: (YYYY-MM-DD, norm_home, norm_away) -> match
    lookup = {}
    for m in matches:
        h = normalize_team_name(m["match_hometeam_name"])
        a = normalize_team_name(m["match_awayteam_name"])
        lookup[(m["match_date"], h, a)] = m

    filled = 0
    for idx in missing_df.index:
        row = df.loc[idx]
        dt  = datetime.strptime(row["date"], "%d/%m/%Y %H:%M")
        m   = lookup.get((dt.strftime("%Y-%m-%d"), row["homeTeam"], row["awayTeam"]))
        if not m:
            continue
        stats = m.get("statistics", [])
        if not stats:
            continue
        home_poss = _get_stat(stats, "Ball Possession", "home")
        away_poss = _get_stat(stats, "Ball Possession", "away")
        if home_poss == 0.0 and away_poss == 0.0:
            home_poss = ""
            away_poss = ""
        def _s(v): return str(v) if v != "" else ""
        df.at[idx, "homePossession"]    = _s(home_poss)
        df.at[idx, "awayPossession"]    = _s(away_poss)
        df.at[idx, "homeShots"]         = _s(_get_stat(stats, "Shots Total",   "home"))
        df.at[idx, "awayShots"]         = _s(_get_stat(stats, "Shots Total",   "away"))
        df.at[idx, "homeShotsOnTarget"] = _s(_get_stat(stats, "Shots On Goal", "home"))
        df.at[idx, "awayShotsOnTarget"] = _s(_get_stat(stats, "Shots On Goal", "away"))
        filled += 1

    return df, filled


def append_new_matches(api_key):
    """Fetch new PL results (with stats) and append them to the historical CSV.

    api_key: football-data.org key (used by main.py for upcoming fixtures —
             kept here so the call signature stays the same).
             New match data is pulled from apifootball.com via APIFOOTBALL_KEY.
    """
    apifootball_key = os.getenv("APIFOOTBALL_KEY")
    if not apifootball_key:
        print("APIFOOTBALL_KEY not set in .env — cannot update match data.")
        return

    print("Checking for new Premier League results...")

    if os.path.exists(HISTORICAL_CSV):
        df_hist = pd.read_csv(HISTORICAL_CSV, dtype=str)
    else:
        df_hist = pd.DataFrame(columns=CSV_COLUMNS)

    # Use the later of (last_update, last date in CSV) as cutoff.
    # This prevents re-adding matches that are already present when
    # last_update.txt has drifted ahead of the actual data.
    last_update = read_last_update()
    if not df_hist.empty:
        last_csv_dt   = pd.to_datetime(df_hist["date"], format="%d/%m/%Y %H:%M").max()
        last_csv_date = last_csv_dt.to_pydatetime().replace(tzinfo=timezone.utc)
    else:
        last_csv_date = datetime(2000, 1, 1, tzinfo=timezone.utc)

    cutoff = min(last_update, last_csv_date)
    today  = datetime.now(timezone.utc)

    print(f"  Last update file : {last_update.date()}")
    print(f"  Latest CSV entry : {last_csv_date.date()}")

    if cutoff.date() >= today.date():
        print("  Already up to date.")
    else:
        date_from = cutoff + timedelta(days=1)

        try:
            matches = _fetch_events(apifootball_key, date_from, today)
        except Exception as e:
            print(f"  Error fetching matches: {e}")
            matches = []

        new_rows = []
        for m in matches:
            row = _convert_match(m)
            is_dup = (
                (df_hist["homeTeam"] == row["homeTeam"]) &
                (df_hist["awayTeam"] == row["awayTeam"]) &
                (df_hist["date"]     == row["date"])
            )
            if not df_hist[is_dup].empty:
                continue
            new_rows.append(row)

        if new_rows:
            df_new = pd.DataFrame(new_rows, columns=CSV_COLUMNS)
            df_new.to_csv(HISTORICAL_CSV, mode="a", header=False, index=False)
            print(f"  Added {len(new_rows)} new match(es).")
        else:
            print("  No new matches found.")

    save_last_update()

    # Backfill stats for any rows already in the CSV that are missing them
    df_current = pd.read_csv(HISTORICAL_CSV, dtype=str)
    missing_count = (
        df_current["homePossession"].isna() | (df_current["homePossession"] == "")
    ).sum()
    if missing_count:
        print(f"  {missing_count} row(s) with missing stats — backfilling...")
        df_current, filled = _backfill_missing_stats(df_current, apifootball_key)
        if filled:
            df_current.to_csv(HISTORICAL_CSV, index=False)
            print(f"  Backfilled {filled} row(s).")
        else:
            print("  Could not fill any missing stats (matches may not be in apifootball yet).")

    print("Done.\n")


if __name__ == "__main__":
    load_dotenv()
    append_new_matches(os.getenv("FOOTBALL_API_KEY"))
