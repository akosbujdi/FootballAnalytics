from models import poisson, random_forest

PREDICTION_MODELS = {
    "poisson": poisson.predict,
    "random_forest": random_forest.predict,
}


def simulate_match(model_name, home_team, away_team, df, n_simulations=1000):
    if model_name not in PREDICTION_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    prediction_fn = PREDICTION_MODELS[model_name]

    home_wins = 0
    draws = 0
    away_wins = 0
    score_counts = {}

    for _ in range(n_simulations):
        h, a = prediction_fn(home_team, away_team, df)

        if h > a:
            home_wins += 1
        elif h < a:
            away_wins += 1
        else:
            draws += 1

        score = f"{h}-{a}"
        score_counts[score] = score_counts.get(score, 0) + 1

    top_score = max(score_counts, key=score_counts.get)
    top_percentage = score_counts[top_score] / n_simulations

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
