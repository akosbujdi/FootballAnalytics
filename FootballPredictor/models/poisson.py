# models/poisson.py
import numpy as np
import pandas as pd

RECENCY_BIAS = True

# ── Recency weighting ────────────────────────────────────────────────────────
HALF_LIFE_DAYS = 60          # weight halves every 60 days
MIN_MATCHES_CONFIDENCE = 5   # below this, blend with league average
PROMOTED_BLEND = 0.25        # how much of their own data to trust for a new team


def _recency_weights(dates: pd.Series, half_life_days: int = HALF_LIFE_DAYS) -> np.ndarray:
    """Exponential decay weights — more recent matches count more."""
    reference = dates.max()
    days_ago = (reference - dates).dt.days.clip(lower=0)
    return np.exp(-np.log(2) * days_ago / half_life_days)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.values
    v = values.values
    total = w.sum()
    if total == 0:
        return np.nan
    return float(np.dot(w, v) / total)


# ── Team strength calculation ────────────────────────────────────────────────

def _compute_team_strengths(df: pd.DataFrame):
    """
    Returns a dict of attack/defence strengths for every team, relative to
    the league average.  Each strength is a (value, n_matches) tuple so the
    caller can apply a confidence blend for low-sample teams.
    """
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

        # ── home attack / defence ──
        if len(home_rows):
            hw = _recency_weights(home_rows['date']) if RECENCY_BIAS else pd.Series(np.ones(len(home_rows)))
            home_att_raw  = _weighted_mean(home_rows['homeGoals'], hw)  # goals scored at home
            home_def_raw  = _weighted_mean(home_rows['awayGoals'], hw)  # goals conceded at home
        else:
            home_att_raw = home_def_raw = np.nan

        # ── away attack / defence ──
        if len(away_rows):
            aw = _recency_weights(away_rows['date']) if RECENCY_BIAS else pd.Series(np.ones(len(away_rows)))
            away_att_raw  = _weighted_mean(away_rows['awayGoals'], aw)  # goals scored away
            away_def_raw  = _weighted_mean(away_rows['homeGoals'], aw)  # goals conceded away
        else:
            away_att_raw = away_def_raw = np.nan

        n_home = len(home_rows)
        n_away = len(away_rows)
        n_total = n_home + n_away

        # Combine home + away into overall attack / defence strengths
        # relative to the league average (ratio form — standard Poisson approach)
        if n_total == 0:
            att_strength = 1.0
            def_strength = 1.0
        else:
            # Weighted combination of home & away sides
            all_rows = pd.concat([home_rows, away_rows])
            all_goals_scored  = pd.concat([home_rows['homeGoals'], away_rows['awayGoals']])
            all_goals_conceded = pd.concat([home_rows['awayGoals'], away_rows['homeGoals']])
            all_weights = _recency_weights(pd.concat([home_rows['date'], away_rows['date']])) if RECENCY_BIAS else pd.Series(np.ones(len(all_rows)))

            wmean_scored   = _weighted_mean(all_goals_scored,   all_weights)
            wmean_conceded = _weighted_mean(all_goals_conceded, all_weights)

            raw_att = wmean_scored   / league_avg if league_avg > 0 else 1.0
            raw_def = wmean_conceded / league_avg if league_avg > 0 else 1.0

            # ── Low-sample confidence blend towards neutral (1.0) ──────────────
            # For a promoted / new team with few matches, pull towards league avg
            confidence = min(n_total / MIN_MATCHES_CONFIDENCE, 1.0)
            if n_total < MIN_MATCHES_CONFIDENCE:
                # Promoted team heuristic: trust their data proportionally
                blend = PROMOTED_BLEND + (1 - PROMOTED_BLEND) * confidence
            else:
                blend = 1.0

            att_strength = blend * raw_att + (1 - blend) * 1.0
            def_strength = blend * raw_def + (1 - blend) * 1.0

        strengths[team] = {
            "att": att_strength,
            "def": def_strength,
            "n":   n_total,
        }

    return strengths, league_avg_home, league_avg_away


def _home_advantage_factor(df: pd.DataFrame) -> float:
    """
    Derive home advantage from the data itself — ratio of average home goals
    to average away goals, clamped to a sensible range.
    """
    h = df['homeGoals'].mean()
    a = df['awayGoals'].mean()
    if a == 0:
        return 1.15
    factor = h / a
    return float(np.clip(factor, 1.0, 1.5))   # sanity clamp


# ── Public interface ─────────────────────────────────────────────────────────

# Module-level cache so strengths are computed once per df version
_cache: dict = {}


def _get_strengths(df: pd.DataFrame):
    key = RECENCY_BIAS   # invalidates if df object changes (good enough for CLI)
    if key not in _cache:
        _cache[key] = _compute_team_strengths(df)
    return _cache[key]


def predict(home_team: str, away_team: str, df: pd.DataFrame):
    """
    Returns (home_goals, away_goals) drawn from Poisson distributions whose
    lambdas are built from recency-weighted attack/defence strengths.
    """
    strengths, league_avg_home, league_avg_away = _get_strengths(df)

    # Fallback: completely unknown team gets neutral strength
    default = {"att": 1.0, "def": 1.0, "n": 0}
    hs = strengths.get(home_team, default)
    as_ = strengths.get(away_team, default)

    home_adv = _home_advantage_factor(df)

    # Expected goals (Dixon-Coles style):
    #   λ_home = league_avg_home × home_attack × away_defence × home_advantage
    #   λ_away = league_avg_away × away_attack × home_defence
    lambda_home = league_avg_home * hs["att"] * as_["def"] * home_adv
    lambda_away = league_avg_away * as_["att"] * hs["def"]

    # Clamp lambdas to plausible range
    lambda_home = float(np.clip(lambda_home, 0.3, 6.0))
    lambda_away = float(np.clip(lambda_away, 0.3, 6.0))

    pred_home = np.random.poisson(lambda_home)
    pred_away = np.random.poisson(lambda_away)

    return int(pred_home), int(pred_away)