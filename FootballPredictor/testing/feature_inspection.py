import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import poisson as _spois

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from models import poisson, random_forest, xgboost_model

DATA_PATH = "../data/historical_matches.csv"
HOME_TEAM = "Leeds United FC"
AWAY_TEAM = "Wolverhampton Wanderers FC"

df = pd.read_csv(DATA_PATH)

# games in dataset
home_n = ((df['homeTeam'] == HOME_TEAM) | (df['awayTeam'] == HOME_TEAM)).sum()
away_n = ((df['homeTeam'] == AWAY_TEAM) | (df['awayTeam'] == AWAY_TEAM)).sum()
print(f"\n{HOME_TEAM} vs {AWAY_TEAM}")
print(f"games in dataset:  {HOME_TEAM} = {home_n},  {AWAY_TEAM} = {away_n}")

# poisson internals
strengths, league_avg_home, league_avg_away = poisson._compute_team_strengths(df)
home_adv = poisson._home_advantage_factor(df)
default = {"att": 1.0, "def": 1.0, "n": 0}
hs = strengths.get(HOME_TEAM, default)
as_ = strengths.get(AWAY_TEAM, default)

lambda_home = float(np.clip(league_avg_home * hs["att"] * as_["def"] * home_adv, 0.3, 6.0))
lambda_away = float(np.clip(league_avg_away * as_["att"] * hs["def"], 0.3, 6.0))

# outcome probs from score matrix
g = np.arange(9)
mat = np.outer(_spois.pmf(g, lambda_home), _spois.pmf(g, lambda_away))
pois_p = [np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()]

print(f"\npoisson internals")
print(f"    league avg goals    home={league_avg_home:.3f}  away={league_avg_away:.3f}  home_adv={home_adv:.3f}")
print(f"    {HOME_TEAM:<30} att={hs['att']:.3f}    def={hs['def']:.3f}  n={hs['n']}")
print(f"    {AWAY_TEAM:<30} att={as_['att']:.3f}    def={as_['def']:.3f}  n={as_['n']}")
print(f"    lambda_home={lambda_home:.3f}  lambda_away={lambda_away:.3f}")

# rf + xgboost feature build
rf_model, rf_means = random_forest._train(df)
xgb_model, xgb_means = xgboost_model._train(df)
form = random_forest._current_form(df)
h2h = random_forest._h2h_stats(HOME_TEAM, AWAY_TEAM, df, rf_means)
row_dict = random_forest._build_predict_row(HOME_TEAM, AWAY_TEAM, form, rf_means, h2h)
row = pd.DataFrame([row_dict])[random_forest.FEATURES].fillna(pd.Series(rf_means)).fillna(0)

print(f"\nfeature values (rf / xgboost)")
for feat in random_forest.FEATURES:
    print(f"    {feat:<28} {row[feat].iloc[0]:.4f}")

print(f"\nh2h (last 5 in dataset)")
for k, v in h2h.items():
    print(f"    {k:<28} {v:.4f}")

# all model predictions
rf_p = rf_model.predict_proba(row)[0]
xgb_p = xgb_model.predict_proba(row)[0]
print(f"\n{'model':<20} {'home':>8} {'draw':>8} {'away':>8}")
print(f"{'    poisson':<20} {pois_p[0]:>8.2%} {pois_p[1]:>8.2%} {pois_p[2]:>8.2%}")
print(f"{'    random forest':<20} {rf_p[0]:>8.2%} {rf_p[1]:>8.2%} {rf_p[2]:>8.2%}")
print(f"{'    xgboost':<20} {xgb_p[0]:>8.2%} {xgb_p[1]:>8.2%} {xgb_p[2]:>8.2%}")

# xgboost top feature importances
print(f"\nxgboost top 10 feature importances")
ranked = sorted(zip(random_forest.FEATURES, xgb_model.feature_importances_), key=lambda x: x[1], reverse=True)
for feat, imp in ranked[:10]:
    print(f"    {feat:<28} {imp:.4f}")
