import json
import os
from datetime import datetime
from utils.name_mapping import normalize_team_name

PRED_FILE = "data/predictions.json"


def save_prediction(model_name, home_team, away_team, top_score, fixture_date):
    os.makedirs("data", exist_ok=True)

    # Load existing predictions
    if os.path.exists(PRED_FILE):
        with open(PRED_FILE, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    entry = {
        "model": model_name,
        "prediction_date": datetime.now().isoformat(),
        "fixture_date": fixture_date,
        "home_team": normalize_team_name(home_team),
        "away_team": normalize_team_name(away_team),
        "predicted_score": top_score
    }

    data.append(entry)

    with open(PRED_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print("Prediction saved.")
