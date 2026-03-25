import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Tunable parameters
N_ESTIMATORS = 300
SHORT_WINDOW = 5
LONG_WINDOW = 15

STAT_COLS = [
    'homeGoals', 'awayGoals',
    'homePossession', 'awayPossession',
    'homeShots', 'awayShots',
    'homeShotsOnTarget', 'awayShotsOnTarget',
]

FEATURES = [
    'h5_scored',  'h5_conceded',  'h5_shots',  'h5_sot',  'h5_poss',
    'h15_scored', 'h15_conceded', 'h15_shots', 'h15_sot', 'h15_poss',
    'a5_scored',  'a5_conceded',  'a5_shots',  'a5_sot',  'a5_poss',
    'a15_scored', 'a15_conceded', 'a15_shots', 'a15_sot', 'a15_poss',
]

_cache = {}


def _rolling_mean(series, window):
    return series.shift().rolling(window, min_periods=1).mean()


def build_features(df):
    df = df.sort_values("date").copy()
    for col in STAT_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Home team rolling stats (from their home matches)
    for col, name in [
        ('homeGoals', 'scored'), ('awayGoals', 'conceded'),
        ('homeShots', 'shots'), ('homeShotsOnTarget', 'sot'), ('homePossession', 'poss'),
    ]:
        df[f'h5_{name}']  = df.groupby('homeTeam')[col].transform(lambda x: _rolling_mean(x, SHORT_WINDOW))
        df[f'h15_{name}'] = df.groupby('homeTeam')[col].transform(lambda x: _rolling_mean(x, LONG_WINDOW))

    # Away team rolling stats (from their away matches)
    for col, name in [
        ('awayGoals', 'scored'), ('homeGoals', 'conceded'),
        ('awayShots', 'shots'), ('awayShotsOnTarget', 'sot'), ('awayPossession', 'poss'),
    ]:
        df[f'a5_{name}']  = df.groupby('awayTeam')[col].transform(lambda x: _rolling_mean(x, SHORT_WINDOW))
        df[f'a15_{name}'] = df.groupby('awayTeam')[col].transform(lambda x: _rolling_mean(x, LONG_WINDOW))

    return df


def _train(df):
    df = build_features(df.copy())
    col_means = df[FEATURES].mean()
    X = df[FEATURES].fillna(col_means).fillna(0)
    y_home = df['homeGoals']
    y_away = df['awayGoals']

    home_model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=42)
    away_model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=42)
    home_model.fit(X, y_home)
    away_model.fit(X, y_away)
    return home_model, away_model, col_means


def _get_models(df):
    key = len(df)
    if key not in _cache:
        _cache[key] = _train(df)
    return _cache[key]


def predict(home_team, away_team, df):
    home_model, away_model, col_means = _get_models(df)

    df_feat = build_features(df.copy())
    latest_home = df_feat[df_feat['homeTeam'] == home_team]
    latest_away = df_feat[df_feat['awayTeam'] == away_team]

    if latest_home.empty or latest_away.empty:
        row = pd.DataFrame([col_means])
    else:
        h = latest_home.iloc[-1]
        a = latest_away.iloc[-1]
        home_feats = ['h5_scored', 'h5_conceded', 'h5_shots', 'h5_sot', 'h5_poss',
                      'h15_scored', 'h15_conceded', 'h15_shots', 'h15_sot', 'h15_poss']
        away_feats = ['a5_scored', 'a5_conceded', 'a5_shots', 'a5_sot', 'a5_poss',
                      'a15_scored', 'a15_conceded', 'a15_shots', 'a15_sot', 'a15_poss']
        row = pd.DataFrame([{**h[home_feats].to_dict(), **a[away_feats].to_dict()}])

    row = row[FEATURES].fillna(col_means).fillna(0)

    expected_home = float(np.clip(home_model.predict(row)[0], 0.1, 3.5))
    expected_away = float(np.clip(away_model.predict(row)[0], 0.1, 3.5))

    return int(np.random.poisson(expected_home)), int(np.random.poisson(expected_away))
