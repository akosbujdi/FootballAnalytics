import os
import sys
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from models import poisson, random_forest

DATA_PATH = "../data/historical_matches.csv"

# Days from the end of the dataset held out for testing
TEST_PERIOD_DAYS = 90


def _outcome(h, a):
    return "H" if h > a else ("A" if h < a else "D")


def _eval_poisson(df_train, df_test):
    poisson._cache.clear()
    strengths, league_avg_home, league_avg_away = poisson._compute_team_strengths(df_train)
    home_adv = poisson._home_advantage_factor(df_train)
    default = {"att": 1.0, "def": 1.0}

    rows = []
    for _, match in df_test.iterrows():
        hs  = strengths.get(match["homeTeam"], default)
        as_ = strengths.get(match["awayTeam"], default)

        lh = float(np.clip(league_avg_home * hs["att"] * as_["def"] * home_adv, 0.3, 6.0))
        la = float(np.clip(league_avg_away * as_["att"] * hs["def"], 0.3, 6.0))

        pred_h, pred_a = round(lh), round(la)
        act_h,  act_a  = int(match["homeGoals"]), int(match["awayGoals"])

        rows.append({
            "home": match["homeTeam"], "away": match["awayTeam"],
            "actual": f"{act_h}-{act_a}", "predicted": f"{pred_h}-{pred_a}",
            "exact":   pred_h == act_h and pred_a == act_a,
            "outcome": _outcome(pred_h, pred_a) == _outcome(act_h, act_a),
            "home_ae": abs(pred_h - act_h),
            "away_ae": abs(pred_a - act_a),
        })

    return pd.DataFrame(rows)


def _eval_rf(df_train, df_test):
    random_forest._cache.clear()
    home_model, away_model, col_means = random_forest._train(df_train)
    df_feat = random_forest.build_features(df_train.copy())

    home_feat_cols = [f for f in random_forest.FEATURES if f.startswith("h")]
    away_feat_cols = [f for f in random_forest.FEATURES if f.startswith("a")]

    rows = []
    for _, match in df_test.iterrows():
        lh_rows = df_feat[df_feat["homeTeam"] == match["homeTeam"]]
        la_rows = df_feat[df_feat["awayTeam"] == match["awayTeam"]]

        if lh_rows.empty or la_rows.empty:
            feat_row = pd.DataFrame([col_means])
        else:
            feat_row = pd.DataFrame([{
                **lh_rows.iloc[-1][home_feat_cols].to_dict(),
                **la_rows.iloc[-1][away_feat_cols].to_dict(),
            }])

        feat_row = feat_row[random_forest.FEATURES].fillna(col_means).fillna(0)

        exp_h = float(np.clip(home_model.predict(feat_row)[0], 0.1, 3.5))
        exp_a = float(np.clip(away_model.predict(feat_row)[0], 0.1, 3.5))

        pred_h, pred_a = round(exp_h), round(exp_a)
        act_h,  act_a  = int(match["homeGoals"]), int(match["awayGoals"])

        rows.append({
            "home": match["homeTeam"], "away": match["awayTeam"],
            "actual": f"{act_h}-{act_a}", "predicted": f"{pred_h}-{pred_a}",
            "exact":   pred_h == act_h and pred_a == act_a,
            "outcome": _outcome(pred_h, pred_a) == _outcome(act_h, act_a),
            "home_ae": abs(pred_h - act_h),
            "away_ae": abs(pred_a - act_a),
        })

    return pd.DataFrame(rows)


def _print_summary(name, df_results):
    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  Test matches:          {len(df_results)}")
    print(f"  Exact score accuracy:  {df_results['exact'].mean():.2%}")
    print(f"  Outcome acc (H/D/A):   {df_results['outcome'].mean():.2%}")
    print(f"  MAE home goals:        {df_results['home_ae'].mean():.3f}")
    print(f"  MAE away goals:        {df_results['away_ae'].mean():.3f}")


def main():
    np.random.seed(42)

    df = pd.read_csv(DATA_PATH)
    df["_date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    df = df.sort_values("_date").reset_index(drop=True)

    # Drop rows with missing goals (unplayed fixtures)
    df = df[df["homeGoals"].notna() & (df["homeGoals"] != "")]
    df = df[df["awayGoals"].notna()  & (df["awayGoals"]  != "")]

    split_date = df["_date"].max() - pd.Timedelta(days=TEST_PERIOD_DAYS)
    df_train = df[df["_date"] <= split_date].drop(columns="_date").reset_index(drop=True)
    df_test  = df[df["_date"] >  split_date].drop(columns="_date").reset_index(drop=True)

    print(f"\nTrain: {len(df_train)} matches  |  Test: {len(df_test)} matches")
    print(f"Test period: {df_test['date'].iloc[0]}  ->  {df_test['date'].iloc[-1]}")

    print("\nEvaluating Poisson...")
    _print_summary("Poisson", _eval_poisson(df_train, df_test))

    print("\nEvaluating Random Forest...")
    _print_summary("Random Forest", _eval_rf(df_train, df_test))
    print()


if __name__ == "__main__":
    main()
