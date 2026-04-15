import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 0 = Home win, 1 = Draw, 2 = Away win
FEATURES = [
    'h_f5_scored', 'h_f5_conceded', 'h_f5_pts',
    'h_f10_scored', 'h_f10_conceded', 'h_f10_pts',
    'h_home_scored', 'h_home_conceded', 'h_home_pts',
    'a_f5_scored', 'a_f5_conceded', 'a_f5_pts',
    'a_f10_scored', 'a_f10_conceded', 'a_f10_pts',
    'a_away_scored', 'a_away_conceded', 'a_away_pts',
    'h2h_home_win_rate', 'h2h_home_scored', 'h2h_away_scored',
    'h_season_ppg', 'a_season_ppg',
    'h_f5_sot', 'h_f5_sot_ag', 'h_f5_shots', 'h_f5_poss', 'h_home_sot',
    'a_f5_sot', 'a_f5_sot_ag', 'a_f5_shots', 'a_f5_poss', 'a_away_sot',
]

_cache = {}


def _pts(scored, conceded):
    return np.where(scored > conceded, 3.0, np.where(scored == conceded, 1.0, 0.0))


def _roll(series, window):
    # shifted rolling mean — no current-game leakage
    return series.shift().rolling(window, min_periods=1).mean()


def _col(df, name):
    # safe numeric column fetch — returns zeros if column missing
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce').values
    return np.zeros(len(df))


def _build_long(df):
    # one row per (team, match), home + away interleaved
    hg = pd.to_numeric(df['homeGoals'], errors='coerce').values
    ag = pd.to_numeric(df['awayGoals'], errors='coerce').values
    h_sot = _col(df, 'homeShotsOnTarget')
    a_sot = _col(df, 'awayShotsOnTarget')
    h_sh = _col(df, 'homeShots')
    a_sh = _col(df, 'awayShots')
    h_pos = _col(df, 'homePossession')
    a_pos = _col(df, 'awayPossession')

    h = pd.DataFrame({
        'date': df['date'].values, 'team': df['homeTeam'].values,
        'scored': hg, 'conceded': ag, 'pts': _pts(hg, ag),
        'sot': h_sot, 'sot_ag': a_sot, 'shots': h_sh, 'poss': h_pos, 'gd': hg - ag,
        'orig_idx': df.index.values, 'side': 'home',
    })
    a = pd.DataFrame({
        'date': df['date'].values, 'team': df['awayTeam'].values,
        'scored': ag, 'conceded': hg, 'pts': _pts(ag, hg),
        'sot': a_sot, 'sot_ag': h_sot, 'shots': a_sh, 'poss': a_pos, 'gd': ag - hg,
        'orig_idx': df.index.values, 'side': 'away',
    })
    long = pd.concat([h, a])
    long['_dt'] = pd.to_datetime(long['date'], dayfirst=True, errors='coerce')
    return long.sort_values(['_dt', 'orig_idx']).drop(columns='_dt').reset_index(drop=True)


def build_features(df):
    df = df.copy()
    df['_dt'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.sort_values('_dt').drop(columns='_dt').reset_index(drop=True)
    for col in ('homeGoals', 'awayGoals'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    long = _build_long(df)

    # overall form
    for w, lbl in [(5, 'f5'), (10, 'f10')]:
        for col in ('scored', 'conceded', 'pts'):
            long[f'{lbl}_{col}'] = long.groupby('team')[col].transform(lambda x: _roll(x, w))

    # shots on target, shots, possession (f5 only)
    for col in ('sot', 'sot_ag', 'shots', 'poss'):
        long[f'f5_{col}'] = long.groupby('team')[col].transform(lambda x: _roll(x, 5))

    # season ppg — within-season to avoid cross-season contamination
    _dts = pd.to_datetime(long['date'], dayfirst=True, errors='coerce')
    long['season'] = _dts.dt.year.where(_dts.dt.month >= 8, _dts.dt.year - 1)
    long['season_ppg'] = long.groupby(['team', 'season'])['pts'].transform(
        lambda x: x.expanding().mean().shift()
    )

    overall_cols = (
            [f'{lbl}_{c}' for lbl in ('f5', 'f10') for c in ('scored', 'conceded', 'pts')]
            + ['f5_sot', 'f5_sot_ag', 'f5_shots', 'f5_poss', 'season_ppg']
    )
    home_rows = long[long['side'] == 'home'].set_index('orig_idx').sort_index()
    away_rows = long[long['side'] == 'away'].set_index('orig_idx').sort_index()

    for col in overall_cols:
        df[f'h_{col}'] = home_rows.loc[df.index, col].values
        df[f'a_{col}'] = away_rows.loc[df.index, col].values

    # venue-specific goals form
    for src, stat in [('homeGoals', 'scored'), ('awayGoals', 'conceded')]:
        df[f'h_home_{stat}'] = df.groupby('homeTeam')[src].transform(lambda x: _roll(x, 5))
    df['_hpts'] = _pts(df['homeGoals'].values, df['awayGoals'].values)
    df['h_home_pts'] = df.groupby('homeTeam')['_hpts'].transform(lambda x: _roll(x, 5))
    df.drop(columns='_hpts', inplace=True)

    for src, stat in [('awayGoals', 'scored'), ('homeGoals', 'conceded')]:
        df[f'a_away_{stat}'] = df.groupby('awayTeam')[src].transform(lambda x: _roll(x, 5))
    df['_apts'] = _pts(df['awayGoals'].values, df['homeGoals'].values)
    df['a_away_pts'] = df.groupby('awayTeam')['_apts'].transform(lambda x: _roll(x, 5))
    df.drop(columns='_apts', inplace=True)

    # venue-specific shots on target
    for src, col in [('homeShotsOnTarget', 'h_home_sot'), ('awayShotsOnTarget', 'a_away_sot')]:
        if src in df.columns:
            s = pd.to_numeric(df[src], errors='coerce')
            grp = 'homeTeam' if src.startswith('home') else 'awayTeam'
            df[col] = df.groupby(grp)[src].transform(
                lambda x: pd.to_numeric(x, errors='coerce').shift().rolling(5, min_periods=1).mean()
            )
        else:
            df[col] = 0.0

    # head-to-head
    pair_min = df[['homeTeam', 'awayTeam']].min(axis=1)
    df['_pair'] = list(zip(pair_min, df[['homeTeam', 'awayTeam']].max(axis=1)))
    df['_hif'] = df['homeTeam'] == pair_min
    df['_t1g'] = np.where(df['_hif'], df['homeGoals'], df['awayGoals'])
    df['_t2g'] = np.where(df['_hif'], df['awayGoals'], df['homeGoals'])
    df['_t1r'] = np.where(df['_t1g'] > df['_t2g'], 1.0, np.where(df['_t1g'] == df['_t2g'], 0.5, 0.0))
    df['_s1'] = df.groupby('_pair')['_t1g'].transform(lambda x: _roll(x, 5))
    df['_s2'] = df.groupby('_pair')['_t2g'].transform(lambda x: _roll(x, 5))
    df['_wr'] = df.groupby('_pair')['_t1r'].transform(lambda x: _roll(x, 5))
    df['h2h_home_win_rate'] = np.where(df['_hif'], df['_wr'], 1 - df['_wr'])
    df['h2h_home_scored'] = np.where(df['_hif'], df['_s1'], df['_s2'])
    df['h2h_away_scored'] = np.where(df['_hif'], df['_s2'], df['_s1'])
    df.drop(columns=['_pair', '_hif', '_t1g', '_t2g', '_t1r', '_s1', '_s2', '_wr'], inplace=True)

    # target: 0=H, 1=D, 2=A
    df['_target'] = np.where(df['homeGoals'] > df['awayGoals'], 0,
                             np.where(df['homeGoals'] == df['awayGoals'], 1, 2)).astype(int)
    return df


def _current_form(df):
    # returns {team: {stat: val}} with each team's current form for prediction
    df = df.copy()
    df['_dt'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.sort_values('_dt').drop(columns='_dt')
    for col in ('homeGoals', 'awayGoals'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    long = _build_long(df)

    # overall form — no shift, include all games
    for w, lbl in [(5, 'f5'), (10, 'f10')]:
        for col in ('scored', 'conceded', 'pts'):
            long[f'{lbl}_{col}'] = long.groupby('team')[col].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )

    for col in ('sot', 'sot_ag', 'shots', 'poss'):
        long[f'f5_{col}'] = long.groupby('team')[col].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

    _dts = pd.to_datetime(long['date'], dayfirst=True, errors='coerce')
    long['season'] = _dts.dt.year.where(_dts.dt.month >= 8, _dts.dt.year - 1)
    long['season_ppg'] = long.groupby(['team', 'season'])['pts'].transform(
        lambda x: x.expanding().mean()
    )

    overall_cols = (
            [f'{lbl}_{c}' for lbl in ('f5', 'f10') for c in ('scored', 'conceded', 'pts')]
            + ['f5_sot', 'f5_sot_ag', 'f5_shots', 'f5_poss', 'season_ppg']
    )
    latest_overall = long.groupby('team')[overall_cols].last()

    home_long = long[long['side'] == 'home'].copy()
    away_long = long[long['side'] == 'away'].copy()

    for col in ('scored', 'conceded', 'pts'):
        home_long[f'home_{col}'] = home_long.groupby('team')[col].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        away_long[f'away_{col}'] = away_long.groupby('team')[col].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

    home_long['home_sot'] = home_long.groupby('team')['sot'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    away_long['away_sot'] = away_long.groupby('team')['sot'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    latest_home = home_long.groupby('team')[['home_scored', 'home_conceded', 'home_pts', 'home_sot']].last()
    latest_away = away_long.groupby('team')[['away_scored', 'away_conceded', 'away_pts', 'away_sot']].last()
    combined = latest_overall.join(latest_home, how='outer').join(latest_away, how='outer')
    return combined.to_dict('index')


def _h2h_stats(home_team, away_team, df, col_means):
    hist = df[
        ((df['homeTeam'] == home_team) & (df['awayTeam'] == away_team)) |
        ((df['homeTeam'] == away_team) & (df['awayTeam'] == home_team))
        ].sort_values('date').tail(5)

    if hist.empty:
        return {f: col_means.get(f, 0.0) for f in ('h2h_home_win_rate', 'h2h_home_scored', 'h2h_away_scored')}

    hg = np.where(hist['homeTeam'].values == home_team,
                  hist['homeGoals'].values, hist['awayGoals'].values).astype(float)
    ag = np.where(hist['awayTeam'].values == away_team,
                  hist['awayGoals'].values, hist['homeGoals'].values).astype(float)
    res = np.where(hg > ag, 1.0, np.where(hg == ag, 0.5, 0.0))
    return {
        'h2h_home_win_rate': float(res.mean()),
        'h2h_home_scored': float(hg.mean()),
        'h2h_away_scored': float(ag.mean()),
    }


def _build_predict_row(home_team, away_team, form, col_means, h2h):
    hf = form.get(home_team, {})
    af = form.get(away_team, {})

    def g(d, k, feat): return d.get(k, col_means.get(feat, 0))

    return {
        'h_f5_scored': g(hf, 'f5_scored', 'h_f5_scored'),
        'h_f5_conceded': g(hf, 'f5_conceded', 'h_f5_conceded'),
        'h_f5_pts': g(hf, 'f5_pts', 'h_f5_pts'),
        'h_f10_scored': g(hf, 'f10_scored', 'h_f10_scored'),
        'h_f10_conceded': g(hf, 'f10_conceded', 'h_f10_conceded'),
        'h_f10_pts': g(hf, 'f10_pts', 'h_f10_pts'),
        'h_home_scored': g(hf, 'home_scored', 'h_home_scored'),
        'h_home_conceded': g(hf, 'home_conceded', 'h_home_conceded'),
        'h_home_pts': g(hf, 'home_pts', 'h_home_pts'),
        'a_f5_scored': g(af, 'f5_scored', 'a_f5_scored'),
        'a_f5_conceded': g(af, 'f5_conceded', 'a_f5_conceded'),
        'a_f5_pts': g(af, 'f5_pts', 'a_f5_pts'),
        'a_f10_scored': g(af, 'f10_scored', 'a_f10_scored'),
        'a_f10_conceded': g(af, 'f10_conceded', 'a_f10_conceded'),
        'a_f10_pts': g(af, 'f10_pts', 'a_f10_pts'),
        'a_away_scored': g(af, 'away_scored', 'a_away_scored'),
        'a_away_conceded': g(af, 'away_conceded', 'a_away_conceded'),
        'a_away_pts': g(af, 'away_pts', 'a_away_pts'),
        'h_season_ppg': g(hf, 'season_ppg', 'h_season_ppg'),
        'a_season_ppg': g(af, 'season_ppg', 'a_season_ppg'),
        'h_f5_sot': g(hf, 'f5_sot', 'h_f5_sot'),
        'h_f5_sot_ag': g(hf, 'f5_sot_ag', 'h_f5_sot_ag'),
        'h_f5_shots': g(hf, 'f5_shots', 'h_f5_shots'),
        'h_f5_poss': g(hf, 'f5_poss', 'h_f5_poss'),
        'h_home_sot': g(hf, 'home_sot', 'h_home_sot'),
        'a_f5_sot': g(af, 'f5_sot', 'a_f5_sot'),
        'a_f5_sot_ag': g(af, 'f5_sot_ag', 'a_f5_sot_ag'),
        'a_f5_shots': g(af, 'f5_shots', 'a_f5_shots'),
        'a_f5_poss': g(af, 'f5_poss', 'a_f5_poss'),
        'a_away_sot': g(af, 'away_sot', 'a_away_sot'),
        **h2h,
    }


def _form_lambdas(home_team, away_team, form, df):
    # expected goals from recent form — used for scoreline sampling
    hf = form.get(home_team, {})
    af = form.get(away_team, {})
    avg = float(df['homeGoals'].mean()) if len(df) else 1.3

    def safe(v):
        try:
            f = float(v)
            return f if not np.isnan(f) and f > 0 else avg
        except (TypeError, ValueError):
            return avg

    h_att = safe(hf.get('home_scored') or hf.get('f5_scored'))
    a_def = safe(af.get('away_conceded') or af.get('f5_conceded'))
    a_att = safe(af.get('away_scored') or af.get('f5_scored'))
    h_def = safe(hf.get('home_conceded') or hf.get('f5_conceded'))

    lambda_home = float(np.clip((h_att + a_def) / 2 * 1.1, 0.3, 4.0))
    lambda_away = float(np.clip((a_att + h_def) / 2, 0.3, 4.0))
    return lambda_home, lambda_away


def _train(df):
    df = build_features(df.copy())
    col_means = df[FEATURES].mean().to_dict()
    valid = df[FEATURES + ['_target']].dropna()
    X, y = valid[FEATURES], valid['_target'].astype(int)
    model = RandomForestClassifier(
        n_estimators=250, max_depth=6, min_samples_leaf=5,
        max_features='sqrt', random_state=42, n_jobs=-1,
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
    h2h = _h2h_stats(home_team, away_team, df, col_means)
    row = pd.DataFrame([_build_predict_row(home_team, away_team, form, col_means, h2h)])
    row = row[FEATURES].fillna(pd.Series(col_means)).fillna(0)
    probs = model.predict_proba(row)[0]
    lh, la = _form_lambdas(home_team, away_team, form, df)
    return tuple(float(p) for p in probs) + (lh, la)
