import pandas as pd
import sys
import os

# give root path to  file
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)
from simulate import simulate_match

DATA_PATH = "../data/historical_matches.csv"
TEST_PERIOD_DAYS = 90  # games of last 3 months

# MonteCarlo settings
N_POISSON = 100
N_RF = 10  # RandomForest much slower, hence fewer simulation

# load dataset
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")

# split data into train/test
split_date = df["date"].max() - pd.Timedelta(days=TEST_PERIOD_DAYS)
train_df = df[df["date"] <= split_date].copy()
test_df = df[df["date"] > split_date].copy()

print(f"Training on {len(train_df)} matches, testing on {len(test_df)} matches.")

# evaluate function
def evaluate_model(model_name, n_simulations):
    outcome_correct = 0
    score_correct = 0
    goal_mae = 0
    total_matches = len(test_df)

    for idx, row in test_df.iterrows():
        home = row["homeTeam"]
        away = row["awayTeam"]
        real_home_goals = row["homeGoals"]
        real_away_goals = row["awayGoals"]

        result = simulate_match(
            model_name=model_name,
            home_team=home,
            away_team=away,
            df=train_df,
            n_simulations=n_simulations
        )

        # predicted outcome
        pred_score = result["top_score"]
        pred_home, pred_away = map(int, pred_score.split("-"))

        # Outcome (H/D/A)
        real_outcome = (
            "H" if real_home_goals > real_away_goals else
            "A" if real_home_goals < real_away_goals else
            "D"
        )
        pred_outcome = (
            "H" if pred_home > pred_away else
            "A" if pred_home < pred_away else
            "D"
        )

        if real_outcome == pred_outcome:
            outcome_correct += 1
        if (real_home_goals, real_away_goals) == (pred_home, pred_away):
            score_correct += 1

        # MAE on goals
        goal_mae += abs(real_home_goals - pred_home) + abs(real_away_goals - pred_away)

    outcome_acc = outcome_correct / total_matches
    score_acc = score_correct / total_matches
    avg_goal_mae = goal_mae / (2 * total_matches)

    return {
        "model": model_name,
        "matches": total_matches,
        "outcome_accuracy": outcome_acc,
        "scoreline_accuracy": score_acc,
        "avg_goal_mae": avg_goal_mae
    }


# running evaluation
print("Evaluating Poisson...")
poisson_results = evaluate_model("poisson", n_simulations=N_POISSON)

print("Evaluating Random Forest...")
rf_results = evaluate_model("random_forest", n_simulations=N_RF)

# print out summary
for res in [poisson_results, rf_results]:
    print(f"\nModel: {res['model']}")
    print(f"Matches evaluated: {res['matches']}")
    print(f"Outcome accuracy: {res['outcome_accuracy']:.2%}")
    print(f"Scoreline accuracy: {res['scoreline_accuracy']:.2%}")
    print(f"Mean Absolute Error (per goal): {res['avg_goal_mae']:.2f}")
