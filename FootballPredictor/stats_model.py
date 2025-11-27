import pandas as pd
import numpy as np

historical_matches = pd.read_csv("data/historical_matches.csv")

TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth AFC": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion": "Brighton",
    "Chelsea FC": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Liverpool FC": "Liverpool",
    "Luton Town": "Luton",  # just in case
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Leeds United": "Leeds",
    "Southampton FC": "Southampton",
    "Tottenham Hotspur": "Spurs",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Burnley FC": "Burnley",
    "Sunderland AFC": "Sunderland"
}


def predict_score_poisson(home_team, away_team):
    # Map API names to CSV names
    home_team_csv = TEAM_NAME_MAP.get(home_team, home_team)
    away_team_csv = TEAM_NAME_MAP.get(away_team, away_team)

    # Home team stats
    home_scored = historical_matches[historical_matches['homeTeam'] == home_team_csv]['homeGoals'].mean()
    home_conceded = historical_matches[historical_matches['homeTeam'] == home_team_csv]['awayGoals'].mean()

    # Away team stats
    away_scored = historical_matches[historical_matches['awayTeam'] == away_team_csv]['awayGoals'].mean()
    away_conceded = historical_matches[historical_matches['awayTeam'] == away_team_csv]['homeGoals'].mean()

    # Replace NaN with a fallback (e.g., league average goals)
    league_avg_home = historical_matches['homeGoals'].mean()
    league_avg_away = historical_matches['awayGoals'].mean()

    home_scored = home_scored if not np.isnan(home_scored) else league_avg_home
    home_conceded = home_conceded if not np.isnan(home_conceded) else league_avg_away
    away_scored = away_scored if not np.isnan(away_scored) else league_avg_away
    away_conceded = away_conceded if not np.isnan(away_conceded) else league_avg_home

    # Expected goals
    expected_home = (home_scored + away_conceded) / 2
    expected_away = (away_scored + home_conceded) / 2

    # Poisson prediction
    pred_home = int(np.round(np.random.poisson(expected_home)))
    pred_away = int(np.round(np.random.poisson(expected_away)))

    return pred_home, pred_away
