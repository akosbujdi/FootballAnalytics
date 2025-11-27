import json
import os
import requests
from datetime import datetime, timezone
from stats_model import predict_score_poisson

# load api key from .env
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FOOTBALL_API_KEY")

TEAMS_FILE = os.path.join("config", "teams.json")
CACHE_FILE = "data/fixtures_cache.json"


# loading fixtures from cache
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


# saving fixture to cache
def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# loading teams from .json file
def load_teams():
    with open(TEAMS_FILE, "r") as f:
        teams = json.load(f)
        return teams


# display team menu to console
def display_team_menu(teams):
    sorted_items = sorted(teams.items(), key=lambda x: int(x[0]))

    while True:
        print("Welcome to Football Predictor!")
        print("Select a Premier League Team:")

        for idx, (team_id, team_name) in enumerate(sorted_items, start=1):
            print(f"{idx}. {team_name}")

        try:
            choice = int(input("\nSelect an option: "))
        except ValueError:
            print("\nInvalid input! Try again.\n")
            continue

        if 1 <= choice <= len(sorted_items):
            team_id, team_name = sorted_items[choice - 1]
            return int(team_id), team_name
        else:
            print(f"\nInvalid input! Enter a number between 1 and {len(sorted_items)}.\n")


# loading next fixture from api call for team_id
def get_next_fixture(team_id, api_key):
    url = f"https://api.football-data.org/v4/teams/{team_id}/matches"
    headers = {"X-Auth-Token": api_key}
    params = {"status": "SCHEDULED", "limit": 1}

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    if data['matches']:
        match = data['matches'][0]
        return {
            "home": match['homeTeam']['name'],
            "away": match['awayTeam']['name'],
            "date": match['utcDate']
        }
    return None


# loading next fixture from cache for team_id
def get_cached_fixture(team_id, api_key):
    cache = load_cache()
    team_key = str(team_id)

    if team_key in cache:
        return cache[team_key]

    fixture = get_next_fixture(team_id, api_key)
    if fixture:
        cache[team_key] = fixture
        save_cache(cache)
    return fixture


# checking if cached fixture is outdated
def is_fixture_outdated(fixture):
    fixture_time = datetime.fromisoformat(fixture['date'].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    return fixture_time < now


# function to remove fixtures from cache that is outdated
def clear_outdated_cache():
    cache = load_cache()
    updated = False

    for team_id, fixture in list(cache.items()):
        if is_fixture_outdated(fixture):
            del cache[team_id]
            updated = True

    if updated:
        save_cache(cache)


# prediction menu (after fixture)
def prediction_menu(home_team, away_team):
    while True:
        print("Predict the scoreline using:")
        print("1. Statistical (Poisson)")
        print("2. AI (coming soon...)")

        try:
            choice = int(input("\nSelect an option: "))
        except ValueError:
            print("\nPlease enter a valid number.\n")
            continue

        if choice == 1:
            pred_home, pred_away = predict_score_poisson(home_team, away_team)
            print(f"Predicted score: {home_team} {pred_home} - {pred_away} {away_team}")
            break
        elif choice == 2:
            print("\nAI method not implemented yet. Please choose another option.\n")
        else:
            print(f"\nInvalid choice. Enter 1 or 2.\n")


# main method
def main():
    clear_outdated_cache()
    teams = load_teams()
    team_id, team_name = display_team_menu(teams)

    print(f"\nYou selected: {team_name}")

    fixture = get_cached_fixture(team_id, API_KEY)

    if fixture:
        print(f"Next match: {fixture['home']} vs {fixture['away']} on {fixture['date']}\n")
    else:
        print("No upcoming fixture found for this team.")

    prediction_menu(fixture['home'], fixture['away'])


if __name__ == "__main__":
    main()
