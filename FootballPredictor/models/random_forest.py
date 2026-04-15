import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ── Tunable parameters ─────────────────────────────────────────────────────────
N_ESTIMATORS     = 200
MAX_DEPTH        = 6
MIN_SAMPLES_LEAF = 5
FORM_SHORT       = 5    # rolling window: overall form (short)
FORM_LONG        = 10   # rolling window: overall form (long)
VENUE_FORM       = 5    # rolling window: venue-specific form
H2H_WINDOW       = 5    # last N head-to-head meetings
# ──────────────────────────────────────────────────────────────────────────────

# 0 = Home win, 1 = Draw, 2 = Away win
FEATURES = [
    'h_f5_scored',  'h_f5_conceded',  'h_f5_pts',
    'h_f10_scored', 'h_f10_conceded', 'h_f10_pts',
    'h_home_scored', 'h_home_conceded', 'h_home_pts',
    'a_f5_scored',  'a_f5_conceded',  'a_f5_pts',
    'a_f10_scored', 'a_f10_conceded', 'a_f10_pts',
    'a_away_scored', 'a_away_conceded', 'a_away_pts',
    'h2h_home_win_rate', 'h2h_home_scored', 'h2h_away_scored',
]

_cache = {}


def _pts(scored, conceded):
    return np.where(scored > conceded, 3.0, np.where(scored == conceded, 1.0, 0.0))


def _roll(series, window):
    """Shifted rolling mean — no current-game leakage."""
    return series.shift().rolling(window, min_periods=1).mean()


def _build_long(df):
    """One row per (team, match), home + away interleaved, for cross-venue form."""
    hg = pd.to_numeric(df['homeGoals'], errors='coerce').values
    ag = pd.to_numeric(df['awayGoals'], errors='coerce').values
    h = pd.DataFrame({
        'date': df['date'].values, 'team': df['homeTeam'].values,
        'scored': hg, 'conceded': ag, 'pts': _pts(hg, ag),
        'orig_idx': df.index.values, 'side': 'home',
    })
    a = pd.DataFrame({
        'date': df['date'].values, 'team': df['awayTeam'].values,
        'scored': ag, 'conceded': hg, 'pts': _pts(ag, hg),
        'orig_idx': df.index.values, 'side': 'away',
    })
    return pd.concat([h, a]).sort_values(['date', 'orig_idx']).reset_index(drop=True)


def build_features(df):
    df = df.sort_values('date').reset_index(drop=True)
    for col in ('homeGoals', 'awayGoals'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── Overall form (home + away combined) ────────────────────────────────────
    long = _build_long(df)
    for w, lbl in [(FORM_SHORT, 'f5'), (FORM_LONG, 'f10')]:
        for col in ('scored', 'conceded', 'pts'):
            long[f'{lbl}_{col}'] = long.groupby('team')[col].transform(lambda x: _roll(x, w))

    overall_cols = [f'{lbl}_{c}' for lbl in ('f5', 'f10') for c in ('scored', 'conceded', 'pts')]
    home_rows = long[long['side'] == 'home'].set_index('orig_idx').sort_index()
    away_rows = long[long['side'] == 'away'].set_index('orig_idx').sort_index()

    for col in overall_cols:
        df[f'h_{col}'] = home_rows.loc[df.index, col].values
        df[f'a_{col}'] = away_rows.loc[df.index, col].values

    # ── Venue-specific form ────────────────────────────────────────────────────
    for src, stat in [('homeGoals', 'scored'), ('awayGoals', 'conceded')]:
        df[f'h_home_{stat}'] = df.groupby('homeTeam')[src].transform(lambda x: _roll(x, VENUE_FORM))
    df['_hpts'] = _pts(df['homeGoals'].values, df['awayGoals'].values)
    df['h_home_pts'] = df.groupby('homeTeam')['_hpts'].transform(lambda x: _roll(x, VENUE_FORM))
    df.drop(columns='_hpts', inplace=True)

    for src, stat in [('awayGoals', 'scored'), ('homeGoals', 'conceded')]:
        df[f'a_away_{stat}'] = df.groupby('awayTeam')[src].transform(lambda x: _roll(x, VENUE_FORM))
    df['_apts'] = _pts(df['awayGoals'].values, df['homeGoals'].values)
    df['a_away_pts'] = df.groupby('awayTeam')['_apts'].transform(lambda x: _roll(x, VENUE_FORM))
    df.drop(columns='_apts', inplace=True)

    # ── Head-to-head ────────────────────────────────────────────────────────────
    pair_min = df[['homeTeam', 'awayTeam']].min(axis=1)
    df['_pair'] = list(zip(pair_min, df[['homeTeam', 'awayTeam']].max(axis=1)))
    df['_hif']  = df['homeTeam'] == pair_min
    df['_t1g']  = np.where(df['_hif'], df['homeGoals'], df['awayGoals'])
    df['_t2g']  = np.where(df['_hif'], df['awayGoals'], df['homeGoals'])
    df['_t1r']  = np.where(df['_t1g'] > df['_t2g'], 1.0, np.where(df['_t1g'] == df['_t2g'], 0.5, 0.0))
    df['_s1']   = df.groupby('_pair')['_t1g'].transform(lambda x: _roll(x, H2H_WINDOW))
    df['_s2']   = df.groupby('_pair')['_t2g'].transform(lambda x: _roll(x, H2H_WINDOW))
    df['_wr']   = df.groupby('_pair')['_t1r'].transform(lambda x: _roll(x, H2H_WINDOW))
    df['h2h_home_win_rate'] = np.where(df['_hif'], df['_wr'],  1 - df['_wr'])
    df['h2h_home_scored']   = np.where(df['_hif'], df['_s1'], df['_s2'])
    df['h2h_away_scored']   = np.where(df['_hif'], df['_s2'], df['_s1'])
    df.drop(columns=['_pair', '_hif', '_t1g', '_t2g', '_t1r', '_s1', '_s2', '_wr'], inplace=True)

    # ── Target: 0=H, 1=D, 2=A ──────────────────────────────────────────────────
    df['_target'] = np.where(df['homeGoals'] > df['awayGoals'], 0,
                    np.where(df['homeGoals'] == df['awayGoals'], 1, 2)).astype(int)
    return df


def _current_form(df):
    """Returns {team: {stat: val}} with each team's up-to-date form for prediction."""
    df = df.sort_values('date').copy()
    for col in ('homeGoals', 'awayGoals'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    long = _build_long(df)
    # No shift — include all games to get current form
    for w, lbl in [(FORM_SHORT, 'f5'), (FORM_LONG, 'f10')]:
        for col in ('scored', 'conceded', 'pts'):
            long[f'{lbl}_{col}'] = long.groupby('team')[col].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )

    overall_cols = [f'{lbl}_{c}' for lbl in ('f5', 'f10') for c in ('scored', 'conceded', 'pts')]
    latest_overall = long.groupby('team')[overall_cols].last()

    home_long = long[long['side'] == 'home'].copy()
    away_long = long[long['side'] == 'away'].copy()
    for col in ('scored', 'conceded', 'pts'):
        home_long[f'home_{col}'] = home_long.groupby('team')[col].transform(
            lambda x: x.rolling(VENUE_FORM, min_periods=1).mean()
        )
        away_long[f'away_{col}'] = away_long.groupby('team')[col].transform(
            lambda x: x.rolling(VENUE_FORM, min_periods=1).mean()
        )

    latest_home = home_long.groupby('team')[['home_scored', 'home_conceded', 'home_pts']].last()
    latest_away = away_long.groupby('team')[['away_scored', 'away_conceded', 'away_pts']].last()
    combined = latest_overall.join(latest_home, how='outer').join(latest_away, how='outer')
    return combined.to_dict('index')


def _h2h_stats(home_team, away_team, df, col_means):
    hist = df[
        ((df['homeTeam'] == home_team) & (df['awayTeam'] == away_team)) |
        ((df['homeTeam'] == away_team) & (df['awayTeam'] == home_team))
    ].sort_values('date').tail(H2H_WINDOW)

    if hist.empty:
        return {f: col_means.get(f, 0.0) for f in ('h2h_home_win_rate', 'h2h_home_scored', 'h2h_away_scored')}

    hg = np.where(hist['homeTeam'].values == home_team,
                  hist['homeGoals'].values, hist['awayGoals'].values).astype(float)
    ag = np.where(hist['awayTeam'].values == away_team,
                  hist['awayGoals'].values, hist['homeGoals'].values).astype(float)
    res = np.where(hg > ag, 1.0, np.where(hg == ag, 0.5, 0.0))
    return {
        'h2h_home_win_rate': float(res.mean()),
        'h2h_home_scored':   float(hg.mean()),
        'h2h_away_scored':   float(ag.mean()),
    }


def _build_predict_row(home_team, away_team, form, col_means, h2h):
    hf = form.get(home_team, {})
    af = form.get(away_team, {})
    return {
        'h_f5_scored':     hf.get('f5_scored',    col_means.get('h_f5_scored', 0)),
        'h_f5_conceded':   hf.get('f5_conceded',  col_means.get('h_f5_conceded', 0)),
        'h_f5_pts':        hf.get('f5_pts',        col_means.get('h_f5_pts', 0)),
        'h_f10_scored':    hf.get('f10_scored',   col_means.get('h_f10_scored', 0)),
        'h_f10_conceded':  hf.get('f10_conceded', col_means.get('h_f10_conceded', 0)),
        'h_f10_pts':       hf.get('f10_pts',       col_means.get('h_f10_pts', 0)),
        'h_home_scored':   hf.get('home_scored',   col_means.get('h_home_scored', 0)),
        'h_home_conceded': hf.get('home_conceded', col_means.get('h_home_conceded', 0)),
        'h_home_pts':      hf.get('home_pts',       col_means.get('h_home_pts', 0)),
        'a_f5_scored':     af.get('f5_scored',    col_means.get('a_f5_scored', 0)),
        'a_f5_conceded':   af.get('f5_conceded',  col_means.get('a_f5_conceded', 0)),
        'a_f5_pts':        af.get('f5_pts',        col_means.get('a_f5_pts', 0)),
        'a_f10_scored':    af.get('f10_scored',   col_means.get('a_f10_scored', 0)),
        'a_f10_conceded':  af.get('f10_conceded', col_means.get('a_f10_conceded', 0)),
        'a_f10_pts':       af.get('f10_pts',       col_means.get('a_f10_pts', 0)),
        'a_away_scored':   af.get('away_scored',   col_means.get('a_away_scored', 0)),
        'a_away_conceded': af.get('away_conceded', col_means.get('a_away_conceded', 0)),
        'a_away_pts':      af.get('away_pts',       col_means.get('a_away_pts', 0)),
        **h2h,
    }


def _train(df):
    df = build_features(df.copy())
    col_means = df[FEATURES].mean().to_dict()
    valid = df[FEATURES + ['_target']].dropna()
    X, y = valid[FEATURES], valid['_target'].astype(int)
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF, max_features='sqrt',
        class_weight='balanced', random_state=42, n_jobs=-1,
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
    form = _current_form(df)
    h2h  = _h2h_stats(home_team, away_team, df, col_means)
    row  = pd.DataFrame([_build_predict_row(home_team, away_team, form, col_means, h2h)])
    row  = row[FEATURES].fillna(pd.Series(col_means)).fillna(0)
    probs = model.predict_proba(row)[0]   # [p_home_win, p_draw, p_away_win]
    return tuple(float(p) for p in probs)
