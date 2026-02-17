import os
import pandas as pd
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from utils.name_mapping import normalize_team_name

load_dotenv()

HISTORICAL_CSV = "data/historical_matches.csv"
LAST_UPDATE_FILE = "data/last_update.txt"

API_URL = "https://api.football-data.org/v4/competitions/PL/matches"

# read in last update from .txt file
def read_last_update():
    if not os.path.exists(LAST_UPDATE_FILE):
        return datetime(2000, 1, 1, tzinfo=timezone.utc)

    with open(LAST_UPDATE_FILE, "r") as f:
        ts = f.read().strip()

    try:
        return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    except:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)


# write new last_update to .txt file (last_update.txt)
def save_last_update():
    with open(LAST_UPDATE_FILE, "w") as f:
        f.write(datetime.now(timezone.utc).isoformat())


# return latest past fixtures
def get_api_matches(api_key):
    headers = {"X-Auth-Token": api_key}
    response = requests.get(API_URL, headers=headers)

    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")

    return response.json().get("matches", [])


# ensure correct naming
def convert_to_csv_format(match):
    date = pd.to_datetime(match["utcDate"])

    date_str = date.strftime("%d/%m/%Y %H:%M")
    home = normalize_team_name(match["homeTeam"]["name"])
    away = normalize_team_name(match["awayTeam"]["name"])
    hg = match["score"]["fullTime"]["home"] if match["score"]["fullTime"]["home"] is not None else ""
    ag = match["score"]["fullTime"]["away"] if match["score"]["fullTime"]["away"] is not None else ""

    return (date_str, home, away, hg, ag)


# main logic: get fixtures from last update until current and update dataset
def append_new_matches(api_key):
    print("Checking for new Premier League results...")

    last_update = read_last_update()
    print(f"Last update: {last_update}")

    matches = get_api_matches(api_key)

    if os.path.exists(HISTORICAL_CSV):
        df_hist = pd.read_csv(HISTORICAL_CSV)
    else:
        df_hist = pd.DataFrame(columns=["date", "homeTeam", "awayTeam", "homeGoals", "awayGoals"])

    new_rows = []

    for match in matches:
        if match["status"] != "FINISHED":
            continue

        match_date = pd.to_datetime(match["utcDate"], utc=True)

        if match_date <= last_update:
            continue

        row = convert_to_csv_format(match)

        if not df_hist[
            (df_hist['homeTeam'] == row[1]) &
            (df_hist['awayTeam'] == row[2]) &
            (df_hist['date'] == row[0])
        ].empty:
            continue

        new_rows.append(row)

    if not new_rows:
        print("No new matches to add.\n")
        return

    df_new = pd.DataFrame(new_rows, columns=["date", "homeTeam", "awayTeam", "homeGoals", "awayGoals"])
    df_new.to_csv(HISTORICAL_CSV, mode="a", header=False, index=False)
    print(f"Added {len(df_new)} new matches to {HISTORICAL_CSV}")

    save_last_update()
    print("Updated last update timestamp.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("FOOTBALL_API_KEY")
    append_new_matches(api_key)
