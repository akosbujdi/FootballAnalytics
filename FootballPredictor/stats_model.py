import numpy as np
import pandas as pd
historical_matches_to_use = pd.read_csv("data/historical_matches.csv")

def predict_score_poisson(home_team, away_team, historical_matches):
    # Home team stats
    home_scored = historical_matches[historical_matches['homeTeam'] == home_team]['homeGoals'].mean()
    home_conceded = historical_matches[historical_matches['homeTeam'] == home_team]['awayGoals'].mean()

    # Away team stats
    away_scored = historical_matches[historical_matches['awayTeam'] == away_team]['awayGoals'].mean()
    away_conceded = historical_matches[historical_matches['awayTeam'] == away_team]['homeGoals'].mean()

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

    # Slightly inflate expected goals to allow bolder outcomes
    expected_home *= np.random.uniform(0.95, 1.05)
    expected_away *= np.random.uniform(0.95, 1.05)

    return pred_home, pred_away

PREDICTION_MODELS = {
    "poisson": predict_score_poisson,
}

def simulate_match(model_name, home_team, away_team, n_simulations=10000):
    if model_name not in PREDICTION_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    prediction_fn = PREDICTION_MODELS[model_name]

    home_wins = 0
    draws = 0
    away_wins = 0
    score_counts = {}

    for _ in range(n_simulations):
        h, a = prediction_fn(home_team, away_team, historical_matches_to_use)

        if h > a:
            home_wins += 1
        elif h < a:
            away_wins += 1
        else:
            draws += 1

        # Score distribution
        score = f"{h}-{a}"
        score_counts[score] = score_counts.get(score, 0) + 1

    # Sort by most frequent scoreline
    sorted_scores = sorted(score_counts.items(), key=lambda x: x[1], reverse=True)

    # Top scoreline
    top_score, top_count = sorted_scores[0]
    top_percentage = top_count / n_simulations

    return {
        "model_used": model_name,
        "probabilities": {
            "home_win": home_wins / n_simulations,
            "draw": draws / n_simulations,
            "away_win": away_wins / n_simulations,
        },
        "score_distribution": score_counts,
        "top_score": top_score,
        "top_score_percentage": top_percentage
    }
