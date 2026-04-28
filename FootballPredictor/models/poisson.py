import numpy as np
import pandas as pd

# Recency_BIAS = True uses recency bias in logic else not
RECENCY_BIAS = True

# Tunable parameters
HALF_LIFE_DAYS = 20  # weight halves every N days
MIN_MATCHES_CONFIDENCE = 5  # below this, blend with league average
PROMOTED_BLEND = 0.25  # data trust weight for promoted/new teams


def _recency_weights(dates: pd.Series, half_life_days: int = HALF_LIFE_DAYS) -> np.ndarray:
    # exponential decay weights relative to most recent match
    reference = dates.max()
    days_ago = (reference - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_ago / half_life_days)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    # weighted average, returns nan if total weight is zero
    w = weights.values
    v = values.values
    total = w.sum()
    if total == 0:
        return np.nan
    return float(np.dot(w, v) / total)


def _compute_team_strengths(df: pd.DataFrame):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)

    league_avg_home = df['homeGoals'].mean()
    league_avg_away = df['awayGoals'].mean()
    league_avg = (league_avg_home + league_avg_away) / 2

    teams = set(df['homeTeam'].unique()) | set(df['awayTeam'].unique())
    strengths = {}

    for team in teams:
        home_rows = df[df['homeTeam'] == team].copy()
        away_rows = df[df['awayTeam'] == team].copy()

        n_total = len(home_rows) + len(away_rows)

        if n_total == 0:
            strengths[team] = {"att": 1.0, "def": 1.0, "n": 0}
            continue

        all_goals_scored = pd.concat([home_rows['homeGoals'], away_rows['awayGoals']])
        all_goals_conceded = pd.concat([home_rows['awayGoals'], away_rows['homeGoals']])
        all_dates = pd.concat([home_rows['date'], away_rows['date']])
        all_weights = _recency_weights(all_dates) if RECENCY_BIAS else pd.Series(np.ones(n_total))

        wmean_scored = _weighted_mean(all_goals_scored, all_weights)
        wmean_conceded = _weighted_mean(all_goals_conceded, all_weights)

        raw_att = wmean_scored / league_avg if league_avg > 0 else 1.0
        raw_def = wmean_conceded / league_avg if league_avg > 0 else 1.0

        # Promoted/low-sample teams: blend towards neutral (1.0)
        confidence = min(n_total / MIN_MATCHES_CONFIDENCE, 1.0)
        blend = PROMOTED_BLEND + (1 - PROMOTED_BLEND) * confidence if n_total < MIN_MATCHES_CONFIDENCE else 1.0

        strengths[team] = {
            "att": blend * raw_att + (1 - blend) * 1.0,
            "def": blend * raw_def + (1 - blend) * 1.0,
            "n": n_total,
        }

    return strengths, league_avg_home, league_avg_away


def _home_advantage_factor(df: pd.DataFrame) -> float:
    # ratio of avg home to away goals, clipped to [1.0, 1.5]
    h = df['homeGoals'].mean()
    a = df['awayGoals'].mean()
    if a == 0:
        return 1.15
    return float(np.clip(h / a, 1.0, 1.5))


_cache: dict = {}


def _get_strengths(df: pd.DataFrame):
    # cached wrapper for _compute_team_strengths
    key = (RECENCY_BIAS, len(df))
    if key not in _cache:
        _cache[key] = _compute_team_strengths(df)
    return _cache[key]


def predict(home_team: str, away_team: str, df: pd.DataFrame):
    strengths, league_avg_home, league_avg_away = _get_strengths(df)

    default = {"att": 1.0, "def": 1.0, "n": 0}
    hs = strengths.get(home_team, default)
    as_ = strengths.get(away_team, default)

    home_adv = _home_advantage_factor(df)

    # Dixon-Coles style expected goals
    lambda_home = league_avg_home * hs["att"] * as_["def"] * home_adv
    lambda_away = league_avg_away * as_["att"] * hs["def"]

    lambda_home = float(np.clip(lambda_home, 0.3, 6.0))
    lambda_away = float(np.clip(lambda_away, 0.3, 6.0))

    return int(np.random.poisson(lambda_home)), int(np.random.poisson(lambda_away))
