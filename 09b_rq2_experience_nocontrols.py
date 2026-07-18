"""
09b_rq2_experience_nocontrols.py
RQ2 ROBUSTNESS (diagnostic): the 07 table re-estimated with NO control variables.
Renamed from 07b_ → 09b_ to sit with the other diagnostics (09/09b); the original
07b_ copy is retained in legacy/. Still depends on 07's sample and outputs.

This is a stripped variant of 07_rq2_experience.py. Everything about the table — the
dependent variable, the seven event families, the two lender samples, the three lookback
windows, the fixed-effect stack, the clustering — is identical. The ONLY change is the
right-hand side:

  07  (baseline)   DV ~ 4 test vars + 5 determinants + 21 controls + FE
  09b (this file)  DV ~ 4 test vars                                 + FE

Dropped: ALL five RQ1 determinants (`accounting_policy`, `offbslease`, `num_rating_suppl_all`,
`non_rated_suppl_all`, `relationship_freq`) and all 21 deal/firm controls. The right-hand side is the
four log(1+x) experience test variables and the fixed effects — nothing else.

The question this answers: are the RQ2 experience coefficients an artefact of the control
set, or do they survive on the fixed effects alone?

★ SAMPLE — the one non-obvious choice
────────────────────────────────────────────────────────────────────────────────────
Dropping the controls from the regression would also drop them from the listwise-deletion
rule, so the sample would *grow* (rows that were excluded only because a control was
missing would come back). The coefficients would then differ from 07 for two reasons at
once — different regressors AND different observations — and the robustness check would be
uninterpretable.

SAME_SAMPLE = True (default) therefore holds the estimation sample fixed to 07's: rows must
still have every Model 7 regressor non-missing, the controls just do not enter the
regression. N is identical to 07 column by column, so any coefficient movement is
attributable to the controls alone. This is the clean test.

SAME_SAMPLE = False estimates on the larger natural sample (test variables non-missing only).
Useful as a second-order check on whether the control set was selecting the sample, but not a
like-for-like comparison with 07.

No winsorization is applied, and none is needed: every variable that 07 winsorized
(`offbslease`, the covenant counts, the firm ratios) has been dropped. All that remains
on the right-hand side is the four test variables, which are log(1+x) and were not winsorized
in 07 either.

Input:  data/contracts.parquet       (output of 03_contracts.py + 04_merge.py)
        data/dealscan_raw.parquet    (for lender_parent_id multi-hot FEs)
Output: output/tables/rq2_experience_nocontrols.xlsx  (sheets "36", "24", "12"; 14 cols each)

07_rq2_experience.py and output/tables/rq2_experience.xlsx are NOT touched.

Runtime ≈ 30 min (48 regressions; the FE matrices dominate, so dropping 23 regressors buys
little). The DV and the FE matrices are window-independent and are built once and reused.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR.parent / "data"
OUT_DIR  = REPO_DIR.parent / "output" / "tables"
OUT_FILE = OUT_DIR / "rq2_experience_nocontrols.xlsx"

MERGE_KEYS = ["borrower_id", "tranche_active_date"]

# Hold the estimation sample fixed to 07's Model 7 sample (see the ★ note in the docstring).
SAME_SAMPLE = True

# Dependent variable: reuse the RQ1 "ALL" dummy (1 if any of the five scores > 0).
DV_SHEET  = "ALL"
DV_SCORES = ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
             "claude_VAR_SCORE", "claude_RES_SCORE"]

WINDOWS = ["36", "24", "12"]

# Event families: (prefix, count-column stem(s), grouping root, table label). See 07.
# `stem` may be a LIST for families that combine count columns; build_experience then takes
# the element-wise row-wise MAX. IC = max(auditor, manager) — consolidated to mirror 07
# (was two separate families ic_a/ic_m; now one, so 7 families / 14 columns like 07).
EVENTS = [
    ("aec",     "est_nc_sum_max_aec",    "aec",  "AEC: Accounting estimate changes"),
    ("fr",      "res_sum_adv1_max_fr",   "fr",   "FR: Financial restatements"),
    ("ic",      ["ic_sum_a_ineff_max_ic",
                 "ic_sum_m_ineff_max_ic"], "ic", "IC: Internal control weakness (auditor or manager, max)"),
    ("aqrm_gc", "aqrm_gc_sum_max_aqrm",  "aqrm", "GC: Going concern"),
    ("aqrm_lf", "aqrm_lf_sum_max_aqrm",  "aqrm", "LF: Late filing"),
    ("aqrm_mi", "aqrm_mi_sum_max_aqrm",  "aqrm", "MI: Material impairment"),
    ("sp",      "sp_default_sum_max_sp", "sp",   "SP: S&P default"),
]

LENDER_SAMPLES = [
    ("All lenders",  "a"),
    ("Lead lenders", "l"),
]

EXPERIENCE_BASE = [
    "NonTop5_Unrelated",
    "Top5_Unrelated",
    "NonTop5_Related",
    "Top5_Related",
]

def exp_names(prefix: str) -> list:
    """Internal (prefixed) names of the four test variables for one event family."""
    return [f"{prefix}_{v}" for v in EXPERIENCE_BASE]

EXPERIENCE_DISPLAY = EXPERIENCE_BASE

# ── What changes vs 07 ────────────────────────────────────────────────────────────
# Non-experience regressors kept: NONE. All five RQ1 determinants (accounting_policy,
# offbslease, num_rating_suppl_all, non_rated_suppl_all, relationship_freq) and all 21 controls
# are dropped, so the right-hand side is the four test variables plus the fixed effects.
KEEP_VARS: list = []

# 07's full Model 7 right-hand side. Retained ONLY to reproduce its listwise-deletion rule
# when SAME_SAMPLE is True — these variables do not enter the regression. MUST stay identical
# to 07's DETERMINANTS/CONTROLS or the fixed sample would diverge (rating pair uses _suppl_all;
# includes the two FISD bond controls, so 09b's sample matches 07 column-by-column).
M7_DETERMINANTS = ["accounting_policy", "offbslease", "num_rating_suppl_all", "non_rated_suppl_all",
                   "relationship_freq"]
M7_CONTROLS = [
    "maturity", "log_lender_count", "log_interest", "log_deal_amount", "perf_pricing",
    "fin_covenant_count", "gen_covenant_count", "secured",
    "size", "profitability", "bsfixed", "liabilities", "logage", "btm", "capex",
    "loss", "rand", "divyield",
    "log_bond_count", "bond_proceeds_scaled",
]


# ── Fixed-effect builders (dense; copied from 07) ─────────────────────────────────

def make_industry_year_fe(df: pd.DataFrame, sic_digits: int = 2):
    """Dummies for (SIC-{sic_digits}d industry) × year."""
    sic_str  = df["sic"].fillna(0).astype(int).astype(str).str.zfill(4).str[:sic_digits]
    ind_year = sic_str + "_" + df["contract_year"].astype(str)
    fe       = pd.get_dummies(ind_year, prefix="iy", drop_first=False, dtype=float)
    return fe, int(ind_year.nunique())


def make_borrower_fe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df["borrower_id"].astype("Int64"), prefix="b",
                          drop_first=False, dtype=float)


def make_lender_multi_hot(df: pd.DataFrame, lender_lists: pd.Series) -> pd.DataFrame:
    """One column per unique lender_parent_id appearing in any deal in the sample
    (multi-hot: a tranche with multiple lenders gets a 1 in each lender's column)."""
    unique_lenders = sorted({lid for lst in lender_lists for lid in lst})
    if not unique_lenders:
        return pd.DataFrame(index=df.index)
    data = {
        f"l_{int(lid)}": lender_lists.apply(lambda lst, v=lid: 1.0 if v in lst else 0.0)
        for lid in unique_lenders
    }
    return pd.DataFrame(data, index=df.index)


# ── Design-matrix cleaner (copied from 07) ────────────────────────────────────────

def stabilize_design(X: pd.DataFrame, y: pd.Series, clusters: pd.Series):
    """Remove non-finite rows, constant columns, duplicate columns, singleton FE columns."""
    row_ok = (
        np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
        & np.isfinite(y.to_numpy(dtype=float))
        & clusters.notna().to_numpy()
    )
    n_dropped = int((~row_ok).sum())
    X, y, clusters = X.loc[row_ok], y.loc[row_ok], clusters.loc[row_ok]

    X = X.loc[:, X.std(ddof=0) > 0]       # constants (incl. all-zero)
    X = X.loc[:, ~X.T.duplicated()]        # duplicates
    X = X.loc[:, X.sum(axis=0) != 1]      # singletons

    X = drop_dependent_columns(X)

    print(f"    stabilize: {n_dropped} non-finite rows dropped, {X.shape[1]} cols remaining")
    return X.astype(float), y, clusters


def drop_dependent_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are exact linear combinations of columns to their LEFT.

    Why this exists (07 does not have it).  The FE blocks are complete dummy sets fitted with
    no intercept, so they are linearly dependent by construction: the industry-year dummies sum
    to 1 on every row, and so do the borrower dummies, hence sum(iy) − sum(borrower) = 0.  The
    design is therefore rank-deficient, and statsmodels' default `pinv` path hands it to
    LAPACK's `gesdd` SVD — which on a matrix this size (10,388 × ~4,000) intermittently fails
    to converge outright:  `numpy.linalg.LinAlgError: SVD did not converge`.  07 gets away with
    it; this variant, with 23 fewer regressors, did not (it died on model 7 of the 12-month
    window after 38 successful fits).

    Removing the dependency does NOT change the estimates.  The dropped columns add nothing to
    the column space, so the fitted values, the residuals, R², and every coefficient on a test
    variable are identical in exact arithmetic — this only makes the solve well-posed.  The
    columns are ordered [test variables, then FE], and an unpivoted QR drops a column only when
    it is dependent on those already accepted, so the test variables (which come first, and are
    not in the FE span) can never be the ones dropped.

    R² stays UNCENTERED (`hasconst=False`, no intercept added), so it remains comparable to 07.
    """
    Xv   = X.to_numpy(dtype=float)
    _, R = np.linalg.qr(Xv, mode="reduced")
    diag = np.abs(np.diag(R))
    tol  = diag.max() * max(Xv.shape) * np.finfo(float).eps
    keep = diag > tol

    n_drop = int((~keep).sum())
    if n_drop:
        print(f"    rank: dropped {n_drop} linearly dependent column(s) "
              f"({X.shape[1]} → {int(keep.sum())})")
    return X.loc[:, keep]


def fit_ols_clustered(y: pd.Series, X: pd.DataFrame, clusters: pd.Series):
    model = sm.OLS(y, X, hasconst=False)
    return model.fit(cov_type="cluster",
                     cov_kwds={"groups": clusters, "use_correction": True})


# ── Output formatting (copied from 07) ────────────────────────────────────────────

def _stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1:  return "*"
    return ""


def _build_labels(regressors: list) -> list:
    labels = []
    for v in regressors:
        labels += [v, ""]
    return labels


def model_column(coefs, tvals, pvals, n_obs: int, r2: float, adj_r2: float,
                 fe_counts: dict, fe_label_order: list,
                 master_regressors: list, dv_name: str, sample_label: str,
                 event_label: str) -> list:
    """Return values list: DV name, event family, lender sample, then coef/t-stat rows
    aligned to master_regressors, then footer rows (N, R², Adj. R², one row per FE type)."""
    values = [dv_name, event_label, sample_label]
    for var in master_regressors:
        if var in coefs.index:
            values.append(f"{coefs[var]:.3f}{_stars(pvals[var])}")
            values.append(f"({tvals[var]:.3f})")
        else:
            values.append("")
            values.append("")

    values.append("")                          # blank separator
    values.append(f"{int(n_obs):,}")           # N
    values.append(f"{r2:.4f}")                 # R²
    values.append(f"{adj_r2:.4f}")             # Adj. R²
    for label in fe_label_order:
        values.append(str(fe_counts[label]) if label in fe_counts else "")

    return values


# ── Shared inputs (copied from 07) ────────────────────────────────────────────────

def build_experience(df: pd.DataFrame, prefix: str, stem, grp_root: str,
                     sample_suf: str, window: str) -> pd.DataFrame:
    """Build the four log(1+x) test variables for one event family and lender sample.

    `stem` may be a single string or a LIST of stems; for a list, the event count is the
    element-wise row-wise MAX of the listed columns (used for IC = max of auditor + manager
    weakness counts).

    The observation's event count is assigned to the Top5 column when the winner lender is
    a top-5 lender and to the Non-Top5 column when it is not; the other column gets 0. A
    None grouping means no event occurred, so both columns get 0. Genuinely missing counts
    (NaN — no lead arranger identified) are preserved as NaN in BOTH columns so the row
    drops from the regression rather than being read as a true zero.
    """
    out = {}
    for rel_code, rel_name in [("u", "Unrelated"), ("r", "Related")]:
        suf = f"{rel_code}{sample_suf}"
        if isinstance(stem, (list, tuple)):
            src_cols = [f"{s}_{suf}_{window}" for s in stem]
            # skipna=False → if any listed count is NaN we keep NaN (genuine missingness).
            val = df[src_cols].max(axis=1, skipna=False).astype(float)
        else:
            val = df[f"{stem}_{suf}_{window}"].astype(float)
        grp = df[f"selected_grouping_{grp_root}_{suf}_{window}"]

        # .eq() on the object dtype yields False for None, so None → 0 in both buckets.
        top5    = np.where(grp.eq(True),  val, 0.0)
        nontop5 = np.where(grp.eq(False), val, 0.0)

        # Restore true missingness: np.where above would have mapped NaN counts (whose
        # grouping is None) to 0. Those must stay NaN.
        missing = val.isna().to_numpy()
        top5[missing]    = np.nan
        nontop5[missing] = np.nan

        out[f"{prefix}_NonTop5_{rel_name}"] = np.log1p(nontop5)
        out[f"{prefix}_Top5_{rel_name}"]    = np.log1p(top5)

        print(f"    {suf}_{window}: Top5 nonzero={int(np.nansum(top5 > 0)):,}  "
              f"NonTop5 nonzero={int(np.nansum(nontop5 > 0)):,}  "
              f"missing={int(missing.sum())}")

    return pd.DataFrame(out, index=df.index)[exp_names(prefix)]


def load_lender_lists() -> pd.Series:
    """Build the (borrower_id, tranche_active_date) → [lender_parent_id, …] mapping
    from dealscan raw, for the lender multi-hot FEs."""
    print("\nLoading lender parent IDs from dealscan raw …")
    ds_raw = pd.read_parquet(
        DATA_DIR / "dealscan_raw.parquet",
        columns=["borrower_id", "tranche_active_date", "lender_parent_id"],
    )
    ds_raw["tranche_active_date"] = pd.to_datetime(ds_raw["tranche_active_date"], errors="coerce")
    ds_raw = ds_raw.dropna(subset=["lender_parent_id"])
    ds_raw["lender_parent_id"] = ds_raw["lender_parent_id"].astype(int)
    return (ds_raw.groupby(MERGE_KEYS)["lender_parent_id"]
            .apply(list)
            .rename("lender_ids"))


# ── Estimation ────────────────────────────────────────────────────────────────────

def prepare_sample(df_full: pd.DataFrame, lender_lists: pd.Series):
    """Apply the sample filters, build the DV and the (sample-invariant) FE matrices.
    Identical to 07 — the divergence happens in run_model's regressor list."""
    dv = f"{DV_SHEET}_score_dummy"
    print(f"\n{'#' * 60}\n#  RQ2 NO-CONTROLS — DV = {dv}  |  windows: {', '.join(WINDOWS)}"
          f"\n#  SAME_SAMPLE = {SAME_SAMPLE}\n{'#' * 60}")

    # Sample filters (identical to 05/07)
    df = df_full[df_full["claude_is_debt_contract"] == "Y"].copy()
    print(f"  {len(df):,} rows after keeping claude_is_debt_contract == Y")
    df = df[df[DV_SCORES].notna().all(axis=1)].copy()
    print(f"  {len(df):,} rows after dropping rows missing any claude score")

    # Dependent variable and derived regressors
    df[dv]              = (df[DV_SCORES] > 0).any(axis=1).astype(float)
    df["contract_year"] = df["tranche_active_date"].dt.year

    # accounting_policy: OR of the two ASC-842 LLM scores (missing kept as missing)
    gaap   = df["claude_GAAP_OVERRIDE_SCORE"]
    freeze = df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan

    print(f"\n{dv} distribution: {df[dv].value_counts().sort_index().to_dict()}")

    # ── Fixed effects (dense) ─────────────────────────────────────────────────
    N          = len(df)
    sic_digits = 2
    fe_iy, n_cells = make_industry_year_fe(df, sic_digits)
    while n_cells >= N - 5 and sic_digits > 1:
        sic_digits -= 1
        fe_iy, n_cells = make_industry_year_fe(df, sic_digits)
    X_fe = fe_iy.loc[:, (fe_iy != 0).any(axis=0)]
    print(f"\nFE matrix: {X_fe.shape[1]} columns (Industry×Year, SIC {sic_digits}-digit)")

    fe_bor = make_borrower_fe(df)
    print(f"Borrower FE matrix: {fe_bor.shape[1]} columns")

    df = df.join(lender_lists, on=MERGE_KEYS)
    df["lender_ids"] = df["lender_ids"].apply(lambda x: x if isinstance(x, list) else [])
    n_with_lenders = (df["lender_ids"].apply(len) > 0).sum()
    print(f"  {n_with_lenders:,} / {len(df):,} tranches matched to at least one lender")

    fe_lender = make_lender_multi_hot(df, df["lender_ids"])
    print(f"Lender FE matrix: {fe_lender.shape[1]} columns")

    fe_labels = [
        f"Industry×Year FEs (SIC {sic_digits}-digit)",
        "Borrower FEs",
        "Lender FEs",
    ]
    return df, dv, X_fe, fe_bor, fe_lender, fe_labels


def run_model(df, dv, X_fe, fe_bor, fe_lender, fe_labels, col_name: str,
              event: tuple, sample_label: str, sample_suf: str, window: str) -> list:
    """Estimate the no-controls spec for one event family × lender sample × window."""
    prefix, stem, grp_root, event_label = event
    experience = exp_names(prefix)
    regressors = experience + KEEP_VARS            # ← the whole point: no controls
    IY_LABEL, BOR_LABEL, LEN_LABEL = fe_labels

    print(f"\n{'=' * 60}\n  [{window}mo] Model {col_name}: {prefix} × {sample_label} "
          f"(source suffix _{sample_suf})\n{'=' * 60}")

    print(f"  building {prefix} test variables:")
    df = df.copy()
    df[experience] = build_experience(df, prefix, stem, grp_root, sample_suf, window)

    def _sample_mask(frame, cols):
        finite = np.isfinite(frame[cols].to_numpy(dtype=float)).all(axis=1)
        return (pd.Series(finite, index=frame.index)
                & frame[dv].notna() & frame["gvkey"].notna())

    # SAME_SAMPLE: require every Model 7 regressor non-missing, exactly as 07 does, so the
    # sample is identical column-by-column and only the right-hand side has changed.
    mask_cols = (experience + M7_DETERMINANTS + M7_CONTROLS) if SAME_SAMPLE else regressors
    mask = _sample_mask(df, mask_cols)
    print(f"\n  Estimation sample: {int(mask.sum()):,} "
          f"({'07 Model 7 sample' if SAME_SAMPLE else 'natural — controls not required'})")

    # No winsorization: every variable 07 winsorized has been dropped. All that remains on the
    # right-hand side is the four log(1+x) test variables, which 07 did not winsorize either.
    df_est = df.loc[mask]

    X_full = pd.concat(
        [df_est[regressors].astype(float),
         X_fe.loc[mask], fe_bor.loc[mask], fe_lender.loc[mask]], axis=1)
    X, y_clean, cl_clean = stabilize_design(X_full, df_est[dv], df_est["gvkey"])
    res = fit_ols_clustered(y_clean, X, cl_clean)

    iy_cols = [c for c in X.columns if c.startswith("iy_")]
    b_cols  = [c for c in X.columns if c.startswith("b_")]
    l_cols  = [c for c in X.columns if c.startswith("l_")]
    fe_counts = {IY_LABEL: len(iy_cols), BOR_LABEL: len(b_cols), LEN_LABEL: len(l_cols)}

    print(f"  N = {len(y_clean):,}  |  R² = {res.rsquared:.4f}  |  Adj. R² = {res.rsquared_adj:.4f}")
    print(f"  IY FEs: {len(iy_cols)}  |  Borrower FEs: {len(b_cols)}  |  Lender FEs: {len(l_cols)}")
    print(f"  Unique clusters (gvkey): {cl_clean.nunique():,}")
    for v in experience + KEEP_VARS:
        if v in res.params.index:
            print(f"    {v:<24} {res.params[v]:>8.4f}{_stars(res.pvalues[v]):<3} "
                  f"(t={res.tvalues[v]:.3f})")

    return model_column(
        res.params, res.tvalues, res.pvalues, len(y_clean),
        res.rsquared, res.rsquared_adj, fe_counts, fe_labels,
        regressors, dv, sample_label, event_label,
    )


# ── Main ──────────────────────────────────────────────────────────────────────────

def run():
    print("Loading contracts …")
    df_full = pd.read_parquet(DATA_DIR / "contracts.parquet")
    print(f"  {len(df_full):,} rows × {df_full.shape[1]} cols")

    lender_lists = load_lender_lists()
    df, dv, X_fe, fe_bor, fe_lender, fe_labels = prepare_sample(df_full, lender_lists)

    display_regressors = EXPERIENCE_DISPLAY + KEEP_VARS
    footer_labels      = ["", "N", "R²", "Adj. R²"] + fe_labels

    tables = {}
    for window in WINDOWS:
        print(f"\n{'#' * 60}\n#  LOOKBACK WINDOW: {window} months  "
              f"(worksheet '{window}')\n{'#' * 60}")

        col_data = {}
        col_num  = 0
        for event in EVENTS:
            for sample_label, sample_suf in LENDER_SAMPLES:
                col_num += 1
                col_name = f"({col_num})"
                col_data[col_name] = run_model(
                    df, dv, X_fe, fe_bor, fe_lender, fe_labels,
                    col_name, event, sample_label, sample_suf, window,
                )

        full_index = (["Dependent variable",
                       f"Lender event from {window} months",
                       "Lender sample"]
                      + _build_labels(display_regressors) + footer_labels)
        tables[window] = pd.DataFrame(col_data, index=full_index)

    OUT_DIR.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        for window in WINDOWS:
            tables[window].to_excel(xw, sheet_name=window)

    print(f"\nSaved → {OUT_FILE}  (sheets: {', '.join(WINDOWS)}; "
          f"{len(EVENTS) * len(LENDER_SAMPLES)} columns each)")


if __name__ == "__main__":
    run()
