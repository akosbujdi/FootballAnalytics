import numpy as np
import pandas as pd

def predict(home_team, away_team, df):

    # Home team stats
    home_scored = df[df['homeTeam'] == home_team]['homeGoals'].mean()

    home_conceded = df[df['homeTeam'] == home_team]['awayGoals'].mean()

    # Away team stats
    away_scored = df[df['awayTeam'] == away_team]['awayGoals'].mean()
    away_conceded = df[df['awayTeam'] == away_team]['homeGoals'].mean()

    league_avg_home = df['homeGoals'].mean()
    league_avg_away = df['awayGoals'].mean()

    home_scored = home_scored if not np.isnan(home_scored) else league_avg_home
    home_conceded = home_conceded if not np.isnan(home_conceded) else league_avg_away
    away_scored = away_scored if not np.isnan(away_scored) else league_avg_away
    away_conceded = away_conceded if not np.isnan(away_conceded) else league_avg_home

    expected_home = (home_scored + away_conceded) / 2
    expected_away = (away_scored + home_conceded) / 2

    pred_home = np.random.poisson(expected_home)
    pred_away = np.random.poisson(expected_away)

    return int(pred_home), int(pred_away)