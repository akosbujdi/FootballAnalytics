import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import poisson as _spois

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from models import poisson, random_forest, xgboost_model

DATA_PATH = "../data/historical_matches.csv"
TEST_PERIOD_DAYS = 200
MAX_GOALS = 8
DRAW_THRESHOLD = 0.4  # predict draw if neither H nor A exceeds this confidence


def _outcome(hg, ag):
    return 'H' if hg > ag else ('D' if hg == ag else 'A')


def _most_likely_poisson(lh, la):
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
        rows.append({"actual": _outcome(int(m["homeGoals"]), int(m["awayGoals"])),
                     "predicted": _most_likely_poisson(lh, la)})
    return pd.DataFrame(rows)


def _eval_ensemble(df_train, df_test, draw_threshold=0.0):
    # average rf + xgboost probabilities before predicting
    random_forest._cache.clear()
    xgboost_model._cache.clear()
    rf_model, rf_means = random_forest._train(df_train)
    xgb_model, xgb_means = xgboost_model._train(df_train)
    form = random_forest._current_form(df_train)
    rows = []
    for _, m in df_test.iterrows():
        h2h = random_forest._h2h_stats(m["homeTeam"], m["awayTeam"], df_train, rf_means)
        row_dict = random_forest._build_predict_row(m["homeTeam"], m["awayTeam"], form, rf_means, h2h)
        row = pd.DataFrame([row_dict])[random_forest.FEATURES].fillna(pd.Series(rf_means)).fillna(0)
        probs = (rf_model.predict_proba(row)[0] + xgb_model.predict_proba(row)[0]) / 2
        if max(probs[0], probs[2]) < draw_threshold:
            pred = 'D'
        else:
            pred = ['H', 'D', 'A'][int(np.argmax(probs))]
        rows.append({"actual": _outcome(int(m["homeGoals"]), int(m["awayGoals"])),
                     "predicted": pred})
    return pd.DataFrame(rows)


def _eval_classifier(train_fn, cache, df_train, df_test, draw_threshold=0.0):
    cache.clear()
    model, col_means = train_fn(df_train)
    form = random_forest._current_form(df_train)
    rows = []
    for _, m in df_test.iterrows():
        h2h = random_forest._h2h_stats(m["homeTeam"], m["awayTeam"], df_train, col_means)
        row_dict = random_forest._build_predict_row(m["homeTeam"], m["awayTeam"], form, col_means, h2h)
        row = pd.DataFrame([row_dict])[random_forest.FEATURES].fillna(pd.Series(col_means)).fillna(0)
        probs = model.predict_proba(row)[0]
        # passive draw: if neither H nor A clears threshold, call it a draw
        if max(probs[0], probs[2]) < draw_threshold:
            pred = 'D'
        else:
            pred = ['H', 'D', 'A'][int(np.argmax(probs))]
        rows.append({"actual": _outcome(int(m["homeGoals"]), int(m["awayGoals"])),
                     "predicted": pred})
    return pd.DataFrame(rows)


def _summary(df_r):
    # returns (overall_acc, home_acc, draw_acc, away_acc)
    acc = lambda out: (df_r.loc[df_r['actual'] == out, 'predicted'] == out).mean() if (
            df_r['actual'] == out).any() else float('nan')
    return (df_r['actual'] == df_r['predicted']).mean(), acc('H'), acc('D'), acc('A')


def main():
    df = pd.read_csv(DATA_PATH)
    df["_date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    df = df.sort_values("_date").reset_index(drop=True)
    df = df[df["homeGoals"].notna() & (df["homeGoals"] != "")]
    df = df[df["awayGoals"].notna() & (df["awayGoals"] != "")]

    split_date = df["_date"].max() - pd.Timedelta(days=TEST_PERIOD_DAYS)
    df_train = df[df["_date"] <= split_date].drop(columns="_date").reset_index(drop=True)
    df_test = df[df["_date"] > split_date].drop(columns="_date").reset_index(drop=True)

    print(f"\ntrain: {len(df_train)} matches  |  test: {len(df_test)} matches"
          f"  ({df_test['date'].iloc[0]} -> {df_test['date'].iloc[-1]})")

    # actual distribution in test set
    actuals = pd.Series([_outcome(int(r['homeGoals']), int(r['awayGoals'])) for _, r in df_test.iterrows()])
    dist = actuals.value_counts()
    print(f"\ntest distribution:  "
          + "  ".join(f"{k}={v} ({v / len(actuals):.0%})" for k, v in dist.items()))
    print(f"naive baseline (always H): {(actuals == 'H').mean():.2%}  "
          f"(always {dist.index[1]}): {dist.iloc[1] / len(actuals):.2%}")

    def fmt(v):
        return f"{v:.0%}" if not (isinstance(v, float) and v != v) else "n/a"

    # standard argmax results
    print(f"\n{'model':<22} {'overall':>8} {'home':>8} {'draw':>8} {'away':>8}")
    print(f"  {'(argmax, n=' + str(len(df_test)) + ')':<20} {'acc':>8} {'acc':>8} {'acc':>8} {'acc':>8}")
    results = [
        ("poisson", _eval_poisson(df_train, df_test)),
        ("random forest", _eval_classifier(random_forest._train, random_forest._cache, df_train, df_test)),
        ("xgboost", _eval_classifier(xgboost_model._train, xgboost_model._cache, df_train, df_test)),
        ("ensemble", _eval_ensemble(df_train, df_test)),
    ]
    for name, df_r in results:
        overall, h, d, a = _summary(df_r)
        print(f"  {name:<20} {fmt(overall):>8} {fmt(h):>8} {fmt(d):>8} {fmt(a):>8}")

    # passive draw threshold results
    print(f"\n{'model':<22} {'overall':>8} {'home':>8} {'draw':>8} {'away':>8}")
    print(
        f"  {'(draw<' + str(DRAW_THRESHOLD) + ', n=' + str(len(df_test)) + ')':<20} {'acc':>8} {'acc':>8} {'acc':>8} {'acc':>8}")
    results_passive = [
        ("poisson", _eval_poisson(df_train, df_test)),
        ("random forest",
         _eval_classifier(random_forest._train, random_forest._cache, df_train, df_test, DRAW_THRESHOLD)),
        ("xgboost", _eval_classifier(xgboost_model._train, xgboost_model._cache, df_train, df_test, DRAW_THRESHOLD)),
        ("ensemble", _eval_ensemble(df_train, df_test, DRAW_THRESHOLD)),
    ]
    for name, df_r in results_passive:
        overall, h, d, a = _summary(df_r)
        print(f"  {name:<20} {fmt(overall):>8} {fmt(h):>8} {fmt(d):>8} {fmt(a):>8}")

    print(f"\nnote: draw threshold predicts draw when neither H nor A confidence exceeds {DRAW_THRESHOLD}")
    print(f"  tune DRAW_THRESHOLD at the top of this file to see the tradeoff")


if __name__ == "__main__":
    main()
