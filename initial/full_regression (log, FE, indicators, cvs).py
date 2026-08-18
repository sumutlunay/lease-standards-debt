import pandas as pd
import numpy as np
import ast
import statsmodels.api as sm
from pathlib import Path
import argparse

# ----------------------------- Config -----------------------------
EVENTS = {
    'aec': 'est_nc_sum',
    'fr': 'res_sum_adv1',
    'ic_a': 'ic_sum_a_ineff',
    'ic_m': 'ic_sum_m_ineff',
    'aqrm_gc': 'aqrm_gc_sum',
    'aqrm_lf': 'aqrm_lf_sum',
    'aqrm_mi': 'aqrm_mi_sum',
}
# Selector prefixes used by your "selected_*" columns
SELECTOR_PREFIX = {
    'aec': 'aec',
    'fr': 'fr',
    'ic_a': 'ic',   # note: both ic_a and ic_m use the same selected_*_ic_* columns
    'ic_m': 'ic',
    'aqrm_gc': 'aqrm',  # aqrm_* share selected_*_aqrm_* columns
    'aqrm_lf': 'aqrm',
    'aqrm_mi': 'aqrm',
}
SAMPLES = {
    'all':  {'inside_suf': 'ia', 'outside_suf': 'oa', 'list_col': 'all_lenders_id_list'},
    'lead': {'inside_suf': 'il', 'outside_suf': 'ol', 'list_col': 'lead_arrangers_id_list'},
}

# ----------------------------- Helpers -----------------------------
def parse_id_list(s):
    """Parse stringified list like '[101, 202]' or '101,202' into a list of ints."""
    if pd.isna(s):
        return []
    if isinstance(s, list):
        return [int(x) for x in s]
    try:
        st = str(s).strip()
        if st.startswith('[') and st.endswith(']'):
            return [int(x) for x in ast.literal_eval(st)]
        # fallback comma-separated
        return [int(x.strip()) for x in st.split(',') if x.strip().isdigit()]
    except Exception:
        return []

def make_naics_level(naics_series, digits=6):
    """Return NAICS code truncated to given digits (as string), handling NaN gracefully."""
    def trunc(x):
        if pd.isna(x):
            return np.nan
        s = str(int(x)) if isinstance(x, (int, np.integer)) else str(x)
        s = ''.join(ch for ch in s if ch.isdigit())
        return s[:digits] if s else np.nan
    return naics_series.apply(trunc)

def make_industry_year_fe(df_sub, naics_digits=6):
    ind = make_naics_level(df_sub['naics'], digits=naics_digits)
    ind_year = ind.astype('string') + "_" + df_sub['contract_year'].astype(int).astype('string')
    fe = pd.get_dummies(ind_year, prefix='iy', drop_first=False, dtype=float)
    return fe, int(ind_year.nunique())

def make_borrower_fe(df_sub):
    return pd.get_dummies(df_sub['borrower_id'].astype('Int64'), prefix='b', drop_first=False, dtype=float)

def build_rhs_aa_columns_LOG_IND(df_sub, event_key, sample_key, counts):
    """
    Build AA_* + indicator block, plus LOG[OtherClientN].

    - AA_* are log(1+X) of raw AA values.
    - Ipos_* are 1{raw AA > 0}.
    - OtherClientN is log(1 + (inside_other_count + outside_other_count)).
    """
    root = EVENTS[event_key]
    sel_prefix = SELECTOR_PREFIX[event_key]
    inside_suf = SAMPLES[sample_key]['inside_suf']
    outside_suf = SAMPLES[sample_key]['outside_suf']

    grp_in = f'selected_grouping_{sel_prefix}_{inside_suf}'
    grp_out = f'selected_grouping_{sel_prefix}_{outside_suf}'
    id_in  = f'selected_parent_lender_id_{sel_prefix}_{inside_suf}'
    id_out = f'selected_parent_lender_id_{sel_prefix}_{outside_suf}'
    val_in  = f'{root}_max_{sel_prefix}_{inside_suf}'
    val_out = f'{root}_max_{sel_prefix}_{outside_suf}'

    # Raw AA_* (before log)
    AA_Top5_inside  = np.zeros(len(df_sub))
    AA_Other_inside = np.zeros(len(df_sub))
    AA_Top5_outside = np.zeros(len(df_sub))
    AA_Other_outside= np.zeros(len(df_sub))

    # Inside winner → place value into Top5 or Other
    top5_in_mask   = (df_sub[grp_in] == 1) | (df_sub[grp_in] == 1.0)
    other_in_mask  = (df_sub[grp_in] == 2) | (df_sub[grp_in] == 2.0)
    if top5_in_mask.notna().any():
        m = top5_in_mask.fillna(False)
        AA_Top5_inside[m] = df_sub.loc[m, val_in].astype(float)
    if other_in_mask.notna().any():
        m = other_in_mask.fillna(False)
        AA_Other_inside[m] = df_sub.loc[m, val_in].astype(float)

    # Outside winner
    top5_out_mask   = (df_sub[grp_out] == 1) | (df_sub[grp_out] == 1.0)
    other_out_mask  = (df_sub[grp_out] == 2) | (df_sub[grp_out] == 2.0)
    if top5_out_mask.notna().any():
        m = top5_out_mask.fillna(False)
        AA_Top5_outside[m] = df_sub.loc[m, val_out].astype(float)
    if other_out_mask.notna().any():
        m = other_out_mask.fillna(False)
        AA_Other_outside[m] = df_sub.loc[m, val_out].astype(float)

    # OtherClientN from counts
    counts_map_inside  = counts.set_index('lender_parent_id')['inside_other_count'].to_dict()
    counts_map_outside = counts.set_index('lender_parent_id')['outside_other_count'].to_dict()
    inside_ids  = df_sub[id_in].round().astype('Int64')
    outside_ids = df_sub[id_out].round().astype('Int64')
    other_in  = inside_ids.map(counts_map_inside).fillna(0).astype(float)
    other_out = outside_ids.map(counts_map_outside).fillna(0).astype(float)
    OtherClientN = other_in + other_out

    # Indicators based on RAW AA_*
    ind_cols = {
        'Ipos_AA_Top5_inside':  (AA_Top5_inside  > 0).astype(float),
        'Ipos_AA_Other_inside': (AA_Other_inside > 0).astype(float),
        'Ipos_AA_Top5_outside': (AA_Top5_outside > 0).astype(float),
        'Ipos_AA_Other_outside':(AA_Other_outside> 0).astype(float),
    }

    # Log(1+X) transform for AA_* (names unchanged)
    eps = 1e-12
    AA_Top5_inside  = np.log1p(np.clip(AA_Top5_inside,  -1 + eps, None))
    AA_Other_inside = np.log1p(np.clip(AA_Other_inside, -1 + eps, None))
    AA_Top5_outside = np.log1p(np.clip(AA_Top5_outside, -1 + eps, None))
    AA_Other_outside= np.log1p(np.clip(AA_Other_outside,-1 + eps, None))

    # LOG[OtherClientN] with log(1+x) to handle zeros
    OtherClientN = np.log1p(np.clip(OtherClientN, 0, None))

    rhs = pd.DataFrame({
        'AA_Top5_inside':  AA_Top5_inside,
        'AA_Other_inside': AA_Other_inside,
        'AA_Top5_outside': AA_Top5_outside,
        'AA_Other_outside':AA_Other_outside,
        'OtherClientN':    OtherClientN
    }, index=df_sub.index)

    rhs = pd.concat([rhs, pd.DataFrame(ind_cols, index=rhs.index)], axis=1)
    return rhs

# Discoverable alias
build_rhs_aa_columns_with_indicators = build_rhs_aa_columns_LOG_IND


def make_lender_multi_hot(df_sub, list_col):
    """Multi-hot lender indicators for FE-B (multiple-lender FE)."""
    lenders_series = df_sub[list_col].apply(parse_id_list)
    unique_lenders = sorted({lid for lst in lenders_series for lid in lst})
    if not unique_lenders:
        return pd.DataFrame(index=df_sub.index), 0
    data = {}
    for lid in unique_lenders:
        col = f'l_{lid}'
        data[col] = lenders_series.apply(lambda lst, v=lid: 1.0 if v in lst else 0.0)
    return pd.DataFrame(data, index=df_sub.index), len(unique_lenders)

def make_pair_multi_hot(df_sub, list_col):
    """Multi-hot Borrower×Lender pair indicators for FE-D (multiple pair FE)."""
    lenders_series = df_sub[list_col].apply(parse_id_list)
    data = {}
    for r, (b, lst) in enumerate(zip(df_sub['borrower_id'], lenders_series)):
        b = int(b)
        for l in lst:
            col = f'p_{b}__{l}'
            col_data = data.get(col)
            if col_data is None:
                col_data = np.zeros(len(df_sub), dtype=float)
                data[col] = col_data
            col_data[r] = 1.0
    return (pd.DataFrame(data, index=df_sub.index) if data else pd.DataFrame(index=df_sub.index), len(data))

def fit_clustered_ols(y, X, clusters):
    """OLS with borrower-clustered SEs; no intercept (FEs replace it)."""
    model = sm.OLS(y, X, hasconst=False)
    return model.fit(cov_type='cluster', cov_kwds={'groups': clusters, 'use_correction': True})

def stabilize_design(X, y, clusters, *, drop_singletons=True, verbose=True, block_hint=None):
    """
    Clean up X to avoid SVD failures:
      1) Remove non-finite rows (and align y, clusters)
      2) Drop constant columns (incl. all-zero)
      3) Drop duplicate columns
      4) Drop singleton FE columns (sum == 1)
    """
    info = {}

    # 1) Remove non-finite rows
    row_ok = np.isfinite(X.to_numpy(dtype=float)).all(axis=1) & np.isfinite(y.to_numpy(dtype=float))
    if clusters is not None:
        row_ok &= clusters.notna().to_numpy()
    n_before = len(X)
    X = X.loc[row_ok]
    y = y.loc[row_ok]
    clusters = clusters.loc[row_ok] if clusters is not None else None
    info['rows_dropped_nonfinite'] = int(n_before - len(X))

    # 2) Drop constant columns (incl. all-zero)
    const_mask = X.std(ddof=0) == 0
    const_cols = X.columns[const_mask].tolist()
    if const_cols:
        X = X.drop(columns=const_cols)
    info['const_cols_dropped'] = len(const_cols)

    # 3) Drop duplicate columns
    dup_mask = X.T.duplicated()
    dup_cols = X.columns[dup_mask].tolist()
    if dup_cols:
        X = X.drop(columns=dup_cols)
    info['duplicate_cols_dropped'] = len(dup_cols)

    # 4) Drop singleton FE columns (sum == 1)
    if drop_singletons:
        col_sums = X.sum(axis=0)
        singleton_cols = col_sums.index[(col_sums == 1)].tolist()
        if singleton_cols:
            X = X.drop(columns=singleton_cols)
        info['singleton_cols_dropped'] = len(singleton_cols)

    X = X.astype(float)

    if verbose:
        msg = f"[stabilize_design{'' if block_hint is None else f' {block_hint}'}] " \
              f"rows-∞/NaN dropped={info['rows_dropped_nonfinite']}, " \
              f"const={info['const_cols_dropped']}, dup={info['duplicate_cols_dropped']}, " \
              f"singleton={info.get('singleton_cols_dropped', 0)}, cols_final={X.shape[1]}"
        print(msg)

    return X, y, clusters, info


def build_controls(df_use: pd.DataFrame) -> pd.DataFrame:
    """
    Build the control-variable block:

    LOG[1+at_mil] + prof_ib_at + ppent_at + (dlc_at+dltt_at) +
    obs_lease_at + xbrl_onbs + xbrl_offbs + xbrl_is + xbrl_vl +
    frozen_gaap + renegotiation_for_gaap_change + not_adopt_asc +
    debt_covenant + lease_covenant +
    log[1+tranche_amount_converted] + log[maturity] +
    log[1+interest_rate] + log[all_lenders_count] +
    relationship_freq + unsecured + non_investment_grade + performance_pricing +
    log[1+covenant_count] + max_debt_covenant + min_liquid_covenant

    All naming kept as in your spec (i.e., we overwrite with log-transforms where applicable).
    """
    z = pd.DataFrame(index=df_use.index)

    # LOG[1+at_mil]
    if 'at_mil' in df_use.columns:
        z['at_mil'] = np.log1p(df_use['at_mil'].astype(float).clip(lower=0))

    # Simple level controls (no additional scaling here)
    for col in [
        'prof_ib_at', 'ppent_at', 'obs_lease_at',
        'xbrl_onbs', 'xbrl_offbs', 'xbrl_is', 'xbrl_vl',
        'frozen_gaap', 'renegotiation_for_gaap_change', 'not_adopt_asc',
        'debt_covenant', 'lease_covenant',
        'relationship_freq', 'unsecured', 'non_investment_grade',
        'performance_pricing',
        'max_debt_covenant', 'min_liquid_covenant'
    ]:
        if col in df_use.columns:
            z[col] = df_use[col].astype(float)

    # dlc_at + dltt_at
    if 'dlc_at' in df_use.columns and 'dltt_at' in df_use.columns:
        z['dlc_dltt_at'] = (df_use['dlc_at'].astype(float) + df_use['dltt_at'].astype(float))

    # log[1+tranche_amount_converted]
    if 'tranche_amount_converted' in df_use.columns:
        z['tranche_amount_converted'] = np.log1p(
            df_use['tranche_amount_converted'].astype(float).clip(lower=0)
        )

    # log[maturity] (assume strictly > 0; otherwise set to NaN)
    if 'maturity' in df_use.columns:
        m = df_use['maturity'].astype(float)
        m = m.where(m > 0, np.nan)
        z['maturity'] = np.log(m)

    # log[1+interest_rate]
    if 'interest_rate' in df_use.columns:
        z['interest_rate'] = np.log1p(
            df_use['interest_rate'].astype(float).clip(lower=0)
        )

    # log[all_lenders_count]
    if 'all_lenders_count' in df_use.columns:
        alc = df_use['all_lenders_count'].astype(float)
        alc = alc.where(alc > 0, np.nan)
        z['all_lenders_count'] = np.log(alc)

    # log[1+covenant_count]
    if 'covenant_count' in df_use.columns:
        z['covenant_count'] = np.log1p(
            df_use['covenant_count'].astype(float).clip(lower=0)
        )

    # NOTE: second_or_lower_lien is intentionally *not* included per your instructions.

    return z


# ----------------------------- Runner -----------------------------
def run_all(input_main, input_counts, output_xlsx):
    df = pd.read_csv(input_main)
    counts = pd.read_csv(input_counts)

    # basic hygiene
    if not np.issubdtype(df['contract_year'].dtype, np.integer):
        df['contract_year'] = df['contract_year'].astype(int, errors='ignore')

    results_B, results_D = [], []

    with pd.ExcelWriter(output_xlsx, engine='xlsxwriter') as xw:
        for event_key, root in EVENTS.items():
            for sample_key, sconf in SAMPLES.items():
                inside_suf  = sconf['inside_suf']
                outside_suf = sconf['outside_suf']
                sel_prefix  = SELECTOR_PREFIX[event_key]
                val_in  = f'{root}_max_{sel_prefix}_{inside_suf}'
                val_out = f'{root}_max_{sel_prefix}_{outside_suf}'

                # lead sample: drop rows missing lead winners (≈ same rule as baseline)
                if sample_key == 'lead':
                    df_use = df[df[val_in].notna() & df[val_out].notna()].copy()
                else:
                    df_use = df.copy()

                # RHS: AA block (log + indicators) and control variables
                rhs_aa   = build_rhs_aa_columns_LOG_IND(df_use, event_key, sample_key, counts)
                rhs_ctrl = build_controls(df_use)
                rhs      = pd.concat([rhs_aa, rhs_ctrl], axis=1)

                # Industry×Year FE
                N = len(df_use)
                chosen_digits = 6
                fe_iy, n_cells = make_industry_year_fe(df_use, naics_digits=chosen_digits)
                while n_cells >= N - 5 and chosen_digits > 3:
                    chosen_digits -= 1
                    fe_iy, n_cells = make_industry_year_fe(df_use, naics_digits=chosen_digits)

                clusters = df_use['borrower_id']
                y = df_use['dv'].astype(float)

                # ----------------- Structure B: Industry×Year + Borrower + multi-lender -----------------
                fe_borrower = make_borrower_fe(df_use)
                lender_mh, n_lenders = make_lender_multi_hot(df_use, sconf['list_col'])

                X_B = pd.concat([rhs, fe_iy, fe_borrower, lender_mh], axis=1).fillna(0.0)
                # drop all-zero columns
                X_B = X_B.loc[:, (X_B != 0).any(axis=0)]
                X_B, yB, clustersB, _ = stabilize_design(X_B, y, clusters, drop_singletons=True, block_hint='FE-B')
                res_B = fit_clustered_ols(yB, X_B, clustersB)

                sheet_name_B = f'B_{event_key}_{sample_key}'
                pd.DataFrame({
                    'coef':   res_B.params,
                    'std_err':res_B.bse,
                    't':      res_B.tvalues,
                    'p':      res_B.pvalues
                }).to_excel(xw, sheet_name=sheet_name_B)

                # counts from final X_B
                iy_cols_B = [c for c in X_B.columns if c.startswith('iy_')]
                l_cols_B  = [c for c in X_B.columns if c.startswith('l_')]
                clusters_borrower_B = int(pd.Series(clustersB).nunique())

                results_B.append({
                    'structure': 'B',
                    'event': event_key,
                    'sample': sample_key,
                    'N_raw': int(N),
                    'rows_used': int(X_B.shape[0]),
                    'NAICS_digits': chosen_digits,
                    'Industry×Year_cells': int(len(iy_cols_B)),
                    'unique_lenders_in_FE': int(len(l_cols_B)),
                    'clusters_borrower': clusters_borrower_B,
                    # key AA terms (already in logs)
                    'coef_AA_Top5_inside': res_B.params.get('AA_Top5_inside', np.nan),
                    'coef_AA_Other_inside': res_B.params.get('AA_Other_inside', np.nan),
                    'coef_AA_Top5_outside': res_B.params.get('AA_Top5_outside', np.nan),
                    'coef_AA_Other_outside': res_B.params.get('AA_Other_outside', np.nan),
                    # indicators
                    'coef_Ipos_AA_Top5_inside': res_B.params.get('Ipos_AA_Top5_inside', np.nan),
                    'coef_Ipos_AA_Other_inside': res_B.params.get('Ipos_AA_Other_inside', np.nan),
                    'coef_Ipos_AA_Top5_outside': res_B.params.get('Ipos_AA_Top5_outside', np.nan),
                    'coef_Ipos_AA_Other_outside': res_B.params.get('Ipos_AA_Other_outside', np.nan),
                    # controls (examples; you can add more here if you want in the summary)
                    'coef_OtherClientN': res_B.params.get('OtherClientN', np.nan),
                    'coef_at_mil':       res_B.params.get('at_mil', np.nan),
                    'coef_prof_ib_at':   res_B.params.get('prof_ib_at', np.nan),
                    'coef_ppent_at':     res_B.params.get('ppent_at', np.nan),
                    'coef_tranche_amount_converted': res_B.params.get('tranche_amount_converted', np.nan),
                    'coef_maturity':     res_B.params.get('maturity', np.nan),
                    'coef_interest_rate':res_B.params.get('interest_rate', np.nan),
                    'coef_all_lenders_count': res_B.params.get('all_lenders_count', np.nan),
                    'coef_covenant_count': res_B.params.get('covenant_count', np.nan),
                    'r2': getattr(res_B, 'rsquared', np.nan),
                    'r2_adj': getattr(res_B, 'rsquared_adj', np.nan),
                })

                # ----------------- Structure D: Industry×Year + multi pair (no borrower FE) -------------
                pair_mh, n_pairs = make_pair_multi_hot(df_use, sconf['list_col'])
                X_D = pd.concat([rhs, fe_iy, pair_mh], axis=1).fillna(0.0)
                X_D = X_D.loc[:, (X_D != 0).any(axis=0)]
                X_D, yD, clustersD, _ = stabilize_design(X_D, y, clusters, drop_singletons=True, block_hint='FE-D')
                res_D = fit_clustered_ols(yD, X_D, clustersD)

                sheet_name_D = f'D_{event_key}_{sample_key}'
                pd.DataFrame({
                    'coef':   res_D.params,
                    'std_err':res_D.bse,
                    't':      res_D.tvalues,
                    'p':      res_D.pvalues
                }).to_excel(xw, sheet_name=sheet_name_D)

                # counts from final X_D
                iy_cols_D = [c for c in X_D.columns if c.startswith('iy_')]
                p_cols_D  = [c for c in X_D.columns if c.startswith('p_')]
                clusters_borrower_D = int(pd.Series(clustersD).nunique())

                results_D.append({
                    'structure': 'D',
                    'event': event_key,
                    'sample': sample_key,
                    'N_raw': int(N),
                    'rows_used': int(X_D.shape[0]),
                    'NAICS_digits': chosen_digits,
                    'Industry×Year_cells': int(len(iy_cols_D)),
                    'unique_pairs_in_FE': int(len(p_cols_D)),
                    'clusters_borrower': clusters_borrower_D,
                    # AA terms
                    'coef_AA_Top5_inside': res_D.params.get('AA_Top5_inside', np.nan),
                    'coef_AA_Other_inside': res_D.params.get('AA_Other_inside', np.nan),
                    'coef_AA_Top5_outside': res_D.params.get('AA_Top5_outside', np.nan),
                    'coef_AA_Other_outside': res_D.params.get('AA_Other_outside', np.nan),
                    # indicators
                    'coef_Ipos_AA_Top5_inside': res_D.params.get('Ipos_AA_Top5_inside', np.nan),
                    'coef_Ipos_AA_Other_inside': res_D.params.get('Ipos_AA_Other_inside', np.nan),
                    'coef_Ipos_AA_Top5_outside': res_D.params.get('Ipos_AA_Top5_outside', np.nan),
                    'coef_Ipos_AA_Other_outside': res_D.params.get('Ipos_AA_Other_outside', np.nan),
                    # controls
                    'coef_OtherClientN': res_D.params.get('OtherClientN', np.nan),
                    'coef_at_mil':       res_D.params.get('at_mil', np.nan),
                    'coef_prof_ib_at':   res_D.params.get('prof_ib_at', np.nan),
                    'coef_ppent_at':     res_D.params.get('ppent_at', np.nan),
                    'coef_tranche_amount_converted': res_D.params.get('tranche_amount_converted', np.nan),
                    'coef_maturity':     res_D.params.get('maturity', np.nan),
                    'coef_interest_rate':res_D.params.get('interest_rate', np.nan),
                    'coef_all_lenders_count': res_D.params.get('all_lenders_count', np.nan),
                    'coef_covenant_count': res_D.params.get('covenant_count', np.nan),
                    'r2': getattr(res_D, 'rsquared', np.nan),
                    'r2_adj': getattr(res_D, 'rsquared_adj', np.nan),
                })

        # Summary sheets
        pd.DataFrame(results_B).to_excel(xw, sheet_name='Summary_FE_B', index=False)
        pd.DataFrame(results_D).to_excel(xw, sheet_name='Summary_FE_D', index=False)


# ----------------------------- CLI -----------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--main',   default='full_model_data.csv',
                   help='path to full_model_data.csv (merged AA + CVs)')
    p.add_argument('--counts', default='parent_other_counts.csv',
                   help='path to parent_other_counts.csv')
    p.add_argument('--out',    default='regression_results_cv.xlsx',
                   help='output Excel workbook path')
    args = p.parse_args()
    run_all(args.main, args.counts, args.out)
