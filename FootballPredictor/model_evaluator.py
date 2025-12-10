import numpy as np
import pandas as pd
from stats_model import predict_score_poisson

historical_matches = pd.read_csv("data/historical_matches.csv")

# Ensure timestamp is parsed correctly
if "date" in historical_matches.columns:
    historical_matches["date"] = pd.to_datetime(
        historical_matches["date"],
        format="%d/%m/%Y %H:%M",
        errors="raise"
    )
else:
    raise ValueError("Your CSV must contain a 'timestamp' column.")


def evaluate_model_after_date(historical_matches, date_cutoff):
    # Convert input to pandas Timestamp
    date_cutoff = pd.to_datetime(date_cutoff)

    # Filter matches after that timestamp
    test_set = historical_matches[historical_matches["date"] > date_cutoff].copy()

    if test_set.empty:
        raise ValueError("No matches found after the given timestamp!")

    predictions = []

    for _, row in test_set.iterrows():
        home = row["homeTeam"]
        away = row["awayTeam"]

        pred_home, pred_away = predict_score_poisson(
            home_team=home,
            away_team=away,
            historical_matches=historical_matches
        )

        predictions.append({
            "homeTeam": home,
            "awayTeam": away,
            "actual_home": row["homeGoals"],
            "actual_away": row["awayGoals"],
            "pred_home": pred_home,
            "pred_away": pred_away
        })

    # Merge predictions
    results_df = test_set.assign(
        pred_home=[p["pred_home"] for p in predictions],
        pred_away=[p["pred_away"] for p in predictions]
    )

    # Helper for winner/draw logic
    def outcome(h, a):
        if h > a: return "H"
        if a > h: return "A"
        return "D"

    # Metrics
    exact_score_accuracy = np.mean(
        (results_df["homeGoals"] == results_df["pred_home"]) &
        (results_df["awayGoals"] == results_df["pred_away"])
    )

    actual_outcomes = results_df.apply(lambda r: outcome(r.homeGoals, r.awayGoals), axis=1)
    predicted_outcomes = results_df.apply(lambda r: outcome(r.pred_home, r.pred_away), axis=1)

    winner_accuracy = np.mean(actual_outcomes == predicted_outcomes)

    mae = np.mean(
        abs(results_df["homeGoals"] - results_df["pred_home"]) +
        abs(results_df["awayGoals"] - results_df["pred_away"])
    )

    metrics = {
        "matches_evaluated": len(results_df),
        "exact_score_accuracy": float(exact_score_accuracy),
        "winner_accuracy": float(winner_accuracy),
        "mean_absolute_error": float(mae)
    }

    return metrics, results_df


metrics, results = evaluate_model_after_date(
    historical_matches,
    "2025-11-01 20:00"
)

print(metrics)
print(results)
