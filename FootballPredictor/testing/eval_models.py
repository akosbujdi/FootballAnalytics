import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import poisson as _spois

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from models import poisson, random_forest, xgboost_model

DATA_PATH = "../data/historical_matches.csv"
TEST_PERIOD_DAYS = 90  # days from dataset end held out for testing
MAX_GOALS = 8


def _outcome(hg, ag):
    return 'H' if hg > ag else ('D' if hg == ag else 'A')


# Poisson evaluation
def _most_likely(lh, la):
    g = np.arange(MAX_GOALS + 1)
    mat = np.outer(_spois.pmf(g, lh), _spois.pmf(g, la))
    home_p = np.tril(mat, k=-1).sum()
    draw_p = np.trace(mat)
    away_p = np.triu(mat, k=1).sum()
    return 'H' if home_p >= draw_p and home_p >= away_p else ('D' if draw_p >= away_p else 'A')


def _eval_poisson(df_train, df_test):
    poisson._cache.clear()
    strengths, avg_home, avg_away = poisson._compute_team_strengths(df_train)
    home_adv = poisson._home_advantage_factor(df_train)
    default = {"att": 1.0, "def": 1.0}
    rows = []
    for _, m in df_test.iterrows():
        hs = strengths.get(m["homeTeam"], default)
        as_ = strengths.get(m["awayTeam"], default)
        lh = float(np.clip(avg_home * hs["att"] * as_["def"] * home_adv, 0.3, 6.0))
        la = float(np.clip(avg_away * as_["att"] * hs["def"], 0.3, 6.0))
        rows.append({
            "actual": _outcome(int(m["homeGoals"]), int(m["awayGoals"])),
            "predicted": _most_likely(lh, la),
        })
    return pd.DataFrame(rows)


# Classifier evaluation

def _eval_classifier(train_fn, cache, df_train, df_test):
    cache.clear()
    model, col_means = train_fn(df_train)
    form = random_forest._current_form(df_train)
    rows = []
    for _, m in df_test.iterrows():
        h2h = random_forest._h2h_stats(m["homeTeam"], m["awayTeam"], df_train, col_means)
        row_dict = random_forest._build_predict_row(m["homeTeam"], m["awayTeam"], form, col_means, h2h)
        row = pd.DataFrame([row_dict])[random_forest.FEATURES].fillna(pd.Series(col_means)).fillna(0)
        probs = model.predict_proba(row)[0]  # [p_H, p_D, p_A]
        pred_out = ['H', 'D', 'A'][int(np.argmax(probs))]
        rows.append({
            "actual": _outcome(int(m["homeGoals"]), int(m["awayGoals"])),
            "predicted": pred_out,
        })
    return pd.DataFrame(rows)


# Reporting

def _print_summary(name, df_r):
    n = len(df_r)
    correct = (df_r['actual'] == df_r['predicted']).mean()
    print(f"\n{'=' * 52}")
    print(f"  {name}  (n={n})")
    print(f"{'=' * 52}")
    print(f"  Outcome accuracy:        {correct:.2%}")
    for outcome, label in [('H', 'Home win'), ('D', 'Draw   '), ('A', 'Away win')]:
        mask = df_r['actual'] == outcome
        cnt = mask.sum()
        if cnt > 0:
            cls_acc = (df_r.loc[mask, 'predicted'] == outcome).mean()
            print(f"  {label} ({cnt:3d} actual):   {cls_acc:.2%} correctly predicted")


def _print_baselines(df_test):
    actuals = [_outcome(int(r['homeGoals']), int(r['awayGoals'])) for _, r in df_test.iterrows()]
    sr = pd.Series(actuals)
    print(f"\n{'=' * 52}")
    print(f"  Baselines  (n={len(df_test)})")
    print(f"{'=' * 52}")
    dist = sr.value_counts()
    for k, v in dist.items():
        print(f"  Actual {k}: {v / len(sr):.2%}  ({v} matches)")
    print(f"  Always predict Home win: {(sr == 'H').mean():.2%}")
    most_common = dist.index[0]
    print(f"  Always predict '{most_common}' (most common): {(sr == most_common).mean():.2%}")


# Main

def main():
    df = pd.read_csv(DATA_PATH)
    df["_date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    df = df.sort_values("_date").reset_index(drop=True)
    df = df[df["homeGoals"].notna() & (df["homeGoals"] != "")]
    df = df[df["awayGoals"].notna() & (df["awayGoals"] != "")]

    split_date = df["_date"].max() - pd.Timedelta(days=TEST_PERIOD_DAYS)
    df_train = df[df["_date"] <= split_date].drop(columns="_date").reset_index(drop=True)
    df_test = df[df["_date"] > split_date].drop(columns="_date").reset_index(drop=True)

    print(f"\nTrain: {len(df_train)} matches  |  Test: {len(df_test)} matches")
    print(f"Test period: {df_test['date'].iloc[0]}  ->  {df_test['date'].iloc[-1]}")

    _print_baselines(df_test)

    print("\nEvaluating Poisson...")
    _print_summary("Poisson (statistical baseline)", _eval_poisson(df_train, df_test))

    print("\nEvaluating Random Forest...")
    _print_summary("Random Forest", _eval_classifier(random_forest._train, random_forest._cache, df_train, df_test))

    print("\nEvaluating XGBoost...")
    _print_summary("XGBoost", _eval_classifier(xgboost_model._train, xgboost_model._cache, df_train, df_test))
    print()


if __name__ == "__main__":
    main()
