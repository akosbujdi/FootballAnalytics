from models import poisson, random_forest, xgboost_model

PREDICTION_MODELS = {
    "poisson":       poisson.predict,
    "random_forest": random_forest.predict,
    "xgboost":       xgboost_model.predict,
}

# These models return (p_home, p_draw, p_away) directly — no simulation loop needed
_CLASSIFIER_MODELS = {"random_forest", "xgboost"}


def simulate_match(model_name, home_team, away_team, df, n_simulations=10000):
    if model_name not in PREDICTION_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    prediction_fn = PREDICTION_MODELS[model_name]

    if model_name in _CLASSIFIER_MODELS:
        home_p, draw_p, away_p = prediction_fn(home_team, away_team, df)
        return {
            "model_used": model_name,
            "probabilities": {"home_win": home_p, "draw": draw_p, "away_win": away_p},
            "score_distribution": {},
            "top_score": "N/A",
            "top_score_percentage": max(home_p, draw_p, away_p),
        }

    # Stochastic simulation (Poisson)
    home_wins = draws = away_wins = 0
    score_counts = {}

    for _ in range(n_simulations):
        h, a = prediction_fn(home_team, away_team, df)
        if h > a:       home_wins += 1
        elif h < a:     away_wins += 1
        else:           draws += 1
        score = f"{h}-{a}"
        score_counts[score] = score_counts.get(score, 0) + 1

    top_score = max(score_counts, key=score_counts.get)
    return {
        "model_used": model_name,
        "probabilities": {
            "home_win": home_wins / n_simulations,
            "draw":     draws     / n_simulations,
            "away_win": away_wins / n_simulations,
        },
        "score_distribution": dict(sorted(score_counts.items(), key=lambda x: x[1], reverse=True)),
        "top_score": top_score,
        "top_score_percentage": score_counts[top_score] / n_simulations,
    }
