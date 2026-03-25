import csv
import io
import os
import pandas as pd
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from utils.name_mapping import normalize_team_name

load_dotenv()

HISTORICAL_CSV  = "data/historical_matches.csv"
LAST_UPDATE_FILE = "data/last_update.txt"
LAST_GW_FILE    = "data/last_gw.txt"

GITHUB_BASE = (
    "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main"
    "/data/2025-2026/By%20Tournament/Premier%20League"
)

CSV_COLUMNS = [
    "date", "homeTeam", "awayTeam", "homeGoals", "awayGoals",
    "homePossession", "awayPossession", "homeShots", "awayShots",
    "homeShotsOnTarget", "awayShotsOnTarget",
]


# --- persistence helpers ---

def read_last_update():
    if not os.path.exists(LAST_UPDATE_FILE):
        return datetime(2000, 1, 1, tzinfo=timezone.utc)
    with open(LAST_UPDATE_FILE) as f:
        ts = f.read().strip()
    try:
        return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    except Exception:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)


def save_last_update():
    with open(LAST_UPDATE_FILE, "w") as f:
        f.write(datetime.now(timezone.utc).isoformat())


def read_last_gw():
    if not os.path.exists(LAST_GW_FILE):
        return 1
    with open(LAST_GW_FILE) as f:
        try:
            return int(f.read().strip())
        except Exception:
            return 1


def save_last_gw(gw):
    with open(LAST_GW_FILE, "w") as f:
        f.write(str(gw))


# --- GitHub fetch helpers ---

def _fetch_team_map(scan_gws):
    """Build {team_code: canonical_name} from teams.csv in the highest available GW."""
    for gw in reversed(scan_gws):
        resp = requests.get(f"{GITHUB_BASE}/GW{gw}/teams.csv", timeout=10)
        if resp.status_code != 200:
            continue
        team_map = {}
        for row in csv.DictReader(io.StringIO(resp.text)):
            try:
                code = int(float(row["code"]))
            except (ValueError, TypeError):
                continue
            fotmob = row.get("fotmob_name") or row.get("name", "")
            team_map[code] = normalize_team_name(fotmob.strip())
        if team_map:
            return team_map
    return {}


def _fetch_gw_fixtures(gw):
    """Return parsed rows from GW{n}/fixtures.csv, or None if the file doesn't exist yet."""
    resp = requests.get(f"{GITHUB_BASE}/GW{gw}/fixtures.csv", timeout=10)
    if resp.status_code != 200:
        return None
    return list(csv.DictReader(io.StringIO(resp.text)))


def _as_score(val):
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return ""


def _as_stat(val):
    try:
        return str(float(val)) if val not in ("", None) else ""
    except (ValueError, TypeError):
        return ""


def _convert_fixture(row, team_map):
    """Convert a fixtures.csv row into a CSV row dict."""
    dt = datetime.strptime(row["kickoff_time"], "%Y-%m-%dT%H:%M:%S")

    home_id = int(float(row["home_team"]))
    away_id = int(float(row["away_team"]))

    return {
        "date":               dt.strftime("%d/%m/%Y %H:%M"),
        "homeTeam":           team_map.get(home_id, f"unknown_{home_id}"),
        "awayTeam":           team_map.get(away_id, f"unknown_{away_id}"),
        "homeGoals":          _as_score(row.get("home_score")),
        "awayGoals":          _as_score(row.get("away_score")),
        "homePossession":     _as_stat(row.get("home_possession")),
        "awayPossession":     _as_stat(row.get("away_possession")),
        "homeShots":          _as_stat(row.get("home_total_shots")),
        "awayShots":          _as_stat(row.get("away_total_shots")),
        "homeShotsOnTarget":  _as_stat(row.get("home_shots_on_target")),
        "awayShotsOnTarget":  _as_stat(row.get("away_shots_on_target")),
    }


# --- main entry point ---
def append_new_matches(api_key):
    """Fetch new PL results from the GitHub stats repo and append to the historical CSV.

    api_key: football-data.org key — kept so main.py call signature is unchanged
             (still used there for upcoming fixture lookups).
    """
    print("Checking for new Premier League results...")

    if os.path.exists(HISTORICAL_CSV):
        df_hist = pd.read_csv(HISTORICAL_CSV, dtype=str)
    else:
        df_hist = pd.DataFrame(columns=CSV_COLUMNS)

    # Cutoff: use the earlier of last_update and last CSV entry to catch any
    # drift where last_update.txt moved ahead of the actual data.
    last_update = read_last_update()
    if not df_hist.empty:
        last_csv_dt   = pd.to_datetime(df_hist["date"], format="%d/%m/%Y %H:%M").max()
        last_csv_date = last_csv_dt.to_pydatetime().replace(tzinfo=timezone.utc)
    else:
        last_csv_date = datetime(2000, 1, 1, tzinfo=timezone.utc)

    cutoff = min(last_update, last_csv_date)

    # GW scan window: go back 1 to catch postponed matches, forward 4 for teams
    # that are ahead in the schedule.
    last_gw    = read_last_gw()
    scan_start = max(1, last_gw - 1)
    scan_end   = last_gw + 4
    scan_gws   = list(range(scan_start, scan_end + 1))

    print(f"  Last update: {last_update.date()} | Latest CSV: {last_csv_date.date()}")
    print(f"  Scanning GW{scan_start}–GW{scan_end}...")

    team_map = _fetch_team_map(scan_gws)
    if not team_map:
        print("  Could not fetch team data from GitHub — check connection.")
        return

    new_rows          = []
    highest_gw_seen   = last_gw

    for gw in scan_gws:
        rows = _fetch_gw_fixtures(gw)
        if rows is None:
            continue

        for row in rows:
            if row.get("finished", "").strip().lower() != "true":
                continue

            try:
                dt = datetime.strptime(row["kickoff_time"], "%Y-%m-%dT%H:%M:%S")
            except (ValueError, KeyError):
                continue

            if dt.replace(tzinfo=timezone.utc) <= cutoff:
                continue

            converted = _convert_fixture(row, team_map)

            is_dup = (
                (df_hist["homeTeam"] == converted["homeTeam"]) &
                (df_hist["awayTeam"] == converted["awayTeam"]) &
                (df_hist["date"]     == converted["date"])
            )
            if not df_hist[is_dup].empty:
                continue

            new_rows.append(converted)
            highest_gw_seen = max(highest_gw_seen, gw)

    if new_rows:
        df_new = pd.DataFrame(new_rows, columns=CSV_COLUMNS)
        df_new.to_csv(HISTORICAL_CSV, mode="a", header=False, index=False)
        print(f"  Added {len(new_rows)} new match(es).")
        save_last_gw(highest_gw_seen)
    else:
        print("  No new matches found.")

    save_last_update()
    print("Done.\n")


if __name__ == "__main__":
    load_dotenv()
    append_new_matches(os.getenv("FOOTBALL_API_KEY"))
