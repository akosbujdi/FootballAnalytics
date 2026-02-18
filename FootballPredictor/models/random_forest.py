import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Global models to keep trained instance
HOME_MODEL = None
AWAY_MODEL = None

def build_features(df):
    """
    Adds rolling averages and season averages for home and away teams
    """
    df = df.sort_values("date")

    # Short-term form (last 5 matches)
    df["home_last5_scored"] = df.groupby("homeTeam")["homeGoals"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["home_last5_conceded"] = df.groupby("homeTeam")["awayGoals"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["away_last5_scored"] = df.groupby("awayTeam")["awayGoals"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["away_last5_conceded"] = df.groupby("awayTeam")["homeGoals"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )

    # Longer-term form (last 15 matches)
    df["home_last15_scored"] = df.groupby("homeTeam")["homeGoals"].transform(
        lambda x: x.shift().rolling(15, min_periods=1).mean()
    )
    df["home_last15_conceded"] = df.groupby("homeTeam")["awayGoals"].transform(
        lambda x: x.shift().rolling(15, min_periods=1).mean()
    )
    df["away_last15_scored"] = df.groupby("awayTeam")["awayGoals"].transform(
        lambda x: x.shift().rolling(15, min_periods=1).mean()
    )
    df["away_last15_conceded"] = df.groupby("awayTeam")["homeGoals"].transform(
        lambda x: x.shift().rolling(15, min_periods=1).mean()
    )

    # Season averages (overall team strength)
    home_season = df.groupby("homeTeam")[["homeGoals","awayGoals"]].transform("mean")
    away_season = df.groupby("awayTeam")[["homeGoals","awayGoals"]].transform("mean")

    df["home_season_scored"] = home_season["homeGoals"]
    df["home_season_conceded"] = home_season["awayGoals"]
    df["away_season_scored"] = away_season["awayGoals"]
    df["away_season_conceded"] = away_season["homeGoals"]

    # Drop NaNs (at start of dataset)
    # df = df.dropna()

    return df

def train(df):
    global HOME_MODEL, AWAY_MODEL

    df = build_features(df.copy())

    features = [
        # Short-term
        "home_last5_scored","home_last5_conceded",
        "away_last5_scored","away_last5_conceded",
        # Long-term
        "home_last15_scored","home_last15_conceded",
        "away_last15_scored","away_last15_conceded",
        # Season-level
        "home_season_scored","home_season_conceded",
        "away_season_scored","away_season_conceded"
    ]

    X = df[features]
    y_home = df["homeGoals"]
    y_away = df["awayGoals"]

    HOME_MODEL = RandomForestRegressor(n_estimators=300, random_state=42)
    AWAY_MODEL = RandomForestRegressor(n_estimators=300, random_state=42)

    HOME_MODEL.fit(X, y_home)
    AWAY_MODEL.fit(X, y_away)

def predict(home_team, away_team, df):
    global HOME_MODEL, AWAY_MODEL

    if HOME_MODEL is None:
        train(df)

    df = build_features(df.copy())

    # Get latest row for each team
    latest_home = df[df["homeTeam"] == home_team].iloc[-1]
    latest_away = df[df["awayTeam"] == away_team].iloc[-1]

    X_new = pd.DataFrame([{
        # Short-term
        "home_last5_scored": latest_home["home_last5_scored"],
        "home_last5_conceded": latest_home["home_last5_conceded"],
        "away_last5_scored": latest_away["away_last5_scored"],
        "away_last5_conceded": latest_away["away_last5_conceded"],
        # Long-term
        "home_last15_scored": latest_home["home_last15_scored"],
        "home_last15_conceded": latest_home["home_last15_conceded"],
        "away_last15_scored": latest_away["away_last15_scored"],
        "away_last15_conceded": latest_away["away_last15_conceded"],
        # Season
        "home_season_scored": latest_home["home_season_scored"],
        "home_season_conceded": latest_home["home_season_conceded"],
        "away_season_scored": latest_away["away_season_scored"],
        "away_season_conceded": latest_away["away_season_conceded"],
    }])

    # Predict expected goals
    expected_home = HOME_MODEL.predict(X_new)[0]
    expected_away = AWAY_MODEL.predict(X_new)[0]

    # Clip expected goals to reasonable range
    expected_home = np.clip(expected_home, 0.1, 3.5)
    expected_away = np.clip(expected_away, 0.1, 3.5)

    # Poisson sampling to create stochastic goals
    simulated_home = np.random.poisson(expected_home)
    simulated_away = np.random.poisson(expected_away)

    return simulated_home, simulated_away


