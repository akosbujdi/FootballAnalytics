import os
import sys
import pandas as pd

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from models import random_forest, xgboost_model

DATA_PATH = "../data/historical_matches.csv"
HOME_TEAM = "Leeds United FC"
AWAY_TEAM = "Wolverhampton Wanderers FC"

df = pd.read_csv(DATA_PATH)

rf_model, rf_means = random_forest._train(df)
xgb_model, xgb_means = xgboost_model._train(df)

form = random_forest._current_form(df)
h2h = random_forest._h2h_stats(HOME_TEAM, AWAY_TEAM, df, rf_means)
row_dict = random_forest._build_predict_row(HOME_TEAM, AWAY_TEAM, form, rf_means, h2h)
row = pd.DataFrame([row_dict])[random_forest.FEATURES].fillna(pd.Series(rf_means)).fillna(0)

# feature values fed to the models
print(f"\n{HOME_TEAM} vs {AWAY_TEAM} — feature values\n")
for feat in random_forest.FEATURES:
    print(f"  {feat:<28} {row[feat].iloc[0]:.4f}")

# h2h
print(f"\nh2h (last 5 in dataset)")
for k, v in h2h.items():
    print(f"  {k:<28} {v:.4f}")

# games in dataset per team
home_n = ((df['homeTeam'] == HOME_TEAM) | (df['awayTeam'] == HOME_TEAM)).sum()
away_n = ((df['homeTeam'] == AWAY_TEAM) | (df['awayTeam'] == AWAY_TEAM)).sum()
print(f"\ngames in dataset:  {HOME_TEAM} = {home_n},  {AWAY_TEAM} = {away_n}")

# predictions
rf_p = rf_model.predict_proba(row)[0]
xgb_p = xgb_model.predict_proba(row)[0]
print(f"\n{'model':<20} {'home':>8} {'draw':>8} {'away':>8}")
print(f"{'random forest':<20} {rf_p[0]:>8.2%} {rf_p[1]:>8.2%} {rf_p[2]:>8.2%}")
print(f"{'xgboost':<20} {xgb_p[0]:>8.2%} {xgb_p[1]:>8.2%} {xgb_p[2]:>8.2%}")

# xgboost top feature importances
print(f"\nxgboost top 10 feature importances")
ranked = sorted(zip(random_forest.FEATURES, xgb_model.feature_importances_), key=lambda x: x[1], reverse=True)
for feat, imp in ranked[:10]:
    print(f"  {feat:<28} {imp:.4f}")
