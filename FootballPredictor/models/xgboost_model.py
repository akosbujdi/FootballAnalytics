import pandas as pd
import xgboost as xgb
from models.random_forest import (
    build_features, FEATURES,
    _current_form, _h2h_stats, _build_predict_row,
)

# ── Tunable parameters ─────────────────────────────────────────────────────────
N_ESTIMATORS     = 500
MAX_DEPTH        = 5
LEARNING_RATE    = 0.01
SUBSAMPLE        = 0.8
COLSAMPLE        = 0.8
MIN_CHILD_WEIGHT = 5
REG_LAMBDA       = 2.0
# ──────────────────────────────────────────────────────────────────────────────

_cache = {}


def _train(df):
    df = build_features(df.copy())
    col_means = df[FEATURES].mean().to_dict()
    valid = df[FEATURES + ['_target']].dropna()
    X, y = valid[FEATURES], valid['_target'].astype(int)
    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE, min_child_weight=MIN_CHILD_WEIGHT,
        reg_lambda=REG_LAMBDA, objective='multi:softprob', num_class=3,
        eval_metric='mlogloss', random_state=42, verbosity=0,
    )
    model.fit(X, y)
    return model, col_means


def _get_model(df):
    key = len(df)
    if key not in _cache:
        _cache[key] = _train(df)
    return _cache[key]


def predict(home_team, away_team, df):
    model, col_means = _get_model(df)
    form  = _current_form(df)
    h2h   = _h2h_stats(home_team, away_team, df, col_means)
    row   = pd.DataFrame([_build_predict_row(home_team, away_team, form, col_means, h2h)])
    row   = row[FEATURES].fillna(pd.Series(col_means)).fillna(0)
    probs = model.predict_proba(row)[0]   # [p_home_win, p_draw, p_away_win]
    return tuple(float(p) for p in probs)
