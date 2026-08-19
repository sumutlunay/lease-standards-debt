"""
06_rq1_determinants.py
RQ1: What explains the design of off-balance-sheet lease covenants?

Dependent variables (one worksheet each; see DV_SPECS) — all linear probability models,
each = 1 if ANY listed claude_*_SCORE > 0, else 0:
  SLB     — claude_SLB_SCORE
  SYN     — claude_SYN_SCORE
  OPL     — claude_OPL_SCORE
  VAR-RES — claude_VAR_SCORE OR claude_RES_SCORE
  ALL     — any of SLB / SYN / OPL / VAR / RES
Each sheet reports the same 8 models below. The set of DVs is easy to extend:
add a (sheet_name, [score_columns]) entry to DV_SPECS.

Models (columns), built up incrementally per Sunay's instructions:
  (1) Industry×Year FEs only (2-digit SIC × year); no regressors
  (2) Industry×Year FEs + Borrower FEs; no regressors
  (3) Industry×Year FEs + Borrower FEs + Lender FEs (multi-hot); no regressors
  (4) Industry×Year FEs + Borrower-Lender Pair FEs (multi-hot, sparse); no regressors
  (5) Model 3's FE structure (IY + Borrower + Lender) + the five RQ1 determinants
  (6) Model 4's FE structure (IY + Borrower-Lender Pair) + the five RQ1 determinants
  (7) Model 5 + deal & borrower-level controls (see CONTROLS)
  (8) Model 6 + deal & borrower-level controls (see CONTROLS)

Models 1–4 project the DV onto fixed effects only (no covariates). Models 5–8 add
the determinants (accounting_policy, offbslease, the credit-quality bucket dummies
BB_grade/B_grade/CCC_below/non_rated_suppl_all, relationship_freq — the buckets are
built on the FISD-supplemented, unrestricted "_all" rating from 03 §2c, with ig_grade
the omitted reference); models 7–8 further add 20 deal/firm controls
(incl. log_bond_count and bond_proceeds_scaled, the FISD bond-activity controls).
Non-logged, non-dummy level variables (offbslease + the covenant counts + the
firm ratios + bond_proceeds_scaled; see WINSOR_LEVEL_VARS) are winsorized at 1% both
tails on each model group's own regression sample.

Two determinant samples, each with a row entering only if all its regressors are
non-missing: m5 (5 determinants; models 5–6) and m7 (5 determinants + 21 controls;
models 7–8). m7 is smaller, so models 7–8 run on fewer observations (bond_proceeds_scaled
is NaN where Compustat total assets is missing/≤0, further trimming the m7 sample).

Models 1–3, 5, 7 are dense OLS with SEs clustered by gvkey, using the same design-matrix
rank guard (stabilize_design → drop_dependent_columns) as 07/08, so their SEs and FE-parameter
counts are consistent with the RQ2/RQ3 baselines. Models 4, 6, 8 use the
sparse pair-FE structure: Model 4 is a pure LSQR projection (no regressors); models
6 and 8 absorb the pair FEs via Frisch-Waugh-Lovell and estimate their regressors on
the FE-partialled-out data with clustered SEs (à la 05b). Their clustered SEs use the
same finite-sample dof convention as the dense columns — the (n−1)/(n−K) correction
counts the absorbed FE parameters in K — so all eight columns share one SE convention
(consistent with 07/08). Because the pair FEs saturate the sample (k/n ≈ 0.99), this
leaves little residual dof and inflates those SEs (with no residual dof the correction
is undefined → NaN t-stats), and the Adj. R² of the pair-FE columns is
degenerate/negative; both plain R² and Adj. R² are reported for every column so the
saturation is visible directly. (Models 6 and 8 also compute a Within R², printed to
the console — the standard fit metric for high-dimensional FE models.)

Input:  data/fulldata.parquet       (output of 03_contracts.py + 04_merge.py)
        data/dealscan_raw.parquet    (for lender_parent_id multi-hot FEs)
Output: output/tables/rq1_determinants.xlsx  (one sheet per DV: SLB, SYN, …)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.sparse import csr_matrix, hstack as sp_hstack
from scipy.sparse.linalg import lsqr as sp_lsqr
from scipy.stats import t as t_dist

REPO_DIR = Path(__file__).resolve().parent.parent  # code/ — this script lives in code/exploratory/
DATA_DIR = REPO_DIR.parent / "data"
OUT_DIR  = REPO_DIR.parent / "output" / "tables"
OUT_FILE = OUT_DIR / "rq1_determinants.xlsx"

# Dependent variables: (worksheet name, [raw claude score columns]). Each entry is
# converted to a dummy named "{sheet}_score_dummy" = 1 if ANY listed score > 0, else 0
# (single-column entries reduce to "1 if score > 0"). The full 8-model table is written
# to its own worksheet. All claude_*_SCORE columns share one missingness pattern (jointly
# present for every scored debt contract), so every DV runs on the same sample. Extend by
# adding entries.
DV_SPECS = [
    ("SLB",     ["claude_SLB_SCORE"]),
    ("SYN",     ["claude_SYN_SCORE"]),
    ("OPL",     ["claude_OPL_SCORE"]),
    ("VAR-RES", ["claude_VAR_SCORE", "claude_RES_SCORE"]),
    ("ALL",     ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
                 "claude_VAR_SCORE", "claude_RES_SCORE"]),
]
MERGE_KEYS = ["borrower_id", "tranche_active_date"]

# The RQ1 determinants entering Model 5 (order = output row order):
#   accounting_policy — OR over the two ASC-842 LLM scores (missing kept as missing)
#   offbslease           — off-BS lease intensity (Compustat + XBRL fallback); winsorized 1% both tails
#   CREDIT QUALITY enters as four mutually-exclusive BUCKET dummies rather than the linear
#   0–22 scale. All are built in 03 on num_rating_suppl_all — the S&P rating SUPPLEMENTED with
#   the borrower's most recent FISD bond rating whenever S&P is missing (UNRESTRICTED "_all"
#   version; no bond-issuance recency filter, 03 §2c):
#     ig_grade   — investment grade, BBB- or above (>= 13)  ← OMITTED REFERENCE, not a regressor
#     BB_grade   — BB+ / BB / BB-                  (10–12)
#     B_grade    — B+ / B / B-                     (7–9)
#     CCC_below  — CCC+ and below, incl. CC, C, D  (1–6)
#     non_rated_suppl_all — still unrated after the supplement (== 0)
#   The linear num_rating_suppl_all, the S&P-only num_rating, and the issuance-restricted
#   num_rating_suppl_iss all remain in fulldata.parquet for robustness.
#   relationship_freq    — fraction of the deal's lenders with a prior 36-month borrower relationship
DETERMINANTS = ["accounting_policy", "offbslease", "BB_grade", "B_grade", "CCC_below", "non_rated_suppl_all", "relationship_freq"]

# Deal + borrower-level controls (Models 7–8), in output row order.
#   log_bond_count       — log(1 + bond_issuance_count); FISD public-bond activity (03 §2a). Logged → not winsorized.
#   bond_proceeds_scaled — cumulative bond proceeds / total assets (04 §7a). Level ratio → winsorized; NaN where at≤0.
CONTROLS = [
    "maturity", "log_lender_count", "log_interest", "log_deal_amount", "perf_pricing",
    "fin_covenant_count", "gen_covenant_count", "secured",
    "size", "profitability", "bsfixed", "liabilities", "logage", "btm", "capex",
    "loss", "rand", "divyield",
    "log_bond_count", "bond_proceeds_scaled",
]
# Winsorize (1% both tails) the non-logged, non-dummy variables. offbslease is a
# determinant but also a level variable, so it is winsorized with this set on each
# model's own regression sample.
WINSOR_LEVEL_VARS = [
    "offbslease",
    "fin_covenant_count", "gen_covenant_count",
    "profitability", "bsfixed", "liabilities", "btm", "capex", "rand", "divyield",
    "bond_proceeds_scaled",   # level ratio (proceeds/assets), heavy right skew; log_bond_count is already logged
]


# ── Fixed-effect builder ───────────────────────────────────────────────────────

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


def make_pair_multi_hot_sparse(df: pd.DataFrame, lender_lists: pd.Series):
    """Sparse (borrower_id × lender_parent_id) pair FEs — multi-hot, one column per
    observed borrower-lender pair. Too numerous (~13K cols) for dense storage.
    Returns (csr_matrix [n_rows × n_pairs], col_names list). Mirrors 05b."""
    rows, col_idx = [], []
    pair_index: dict = {}
    n = len(df)
    for row_pos, (b, lst) in enumerate(zip(df["borrower_id"].values, lender_lists.values)):
        b = int(b)
        for lid in lst:
            key = (b, int(lid))
            if key not in pair_index:
                pair_index[key] = len(pair_index)
            rows.append(row_pos)
            col_idx.append(pair_index[key])
    n_cols = len(pair_index)
    if n_cols == 0:
        return csr_matrix((n, 0)), []
    mat = csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, col_idx)),
        shape=(n, n_cols),
    )
    col_names = [f"p_{b}__{l}" for b, l in pair_index]
    return mat, col_names


def _dedup_sparse_cols(mat: csr_matrix) -> csr_matrix:
    """Remove exact-duplicate columns (identical non-zero row-index sets) from a
    sparse matrix via byte-hashing each column's index array. Mirrors 05b."""
    mat_csc = mat.tocsc()
    seen: set = set()
    keep: list = []
    for j in range(mat_csc.shape[1]):
        start = int(mat_csc.indptr[j])
        end   = int(mat_csc.indptr[j + 1])
        key   = mat_csc.indices[start:end].tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(j)
    if len(keep) == mat_csc.shape[1]:
        return mat
    return mat[:, keep]


def fe_only_r2_sparse(y: np.ndarray, X_fe: csr_matrix):
    """Project y onto a sparse FE matrix (LSQR) and return (r2, adj_r2, k).
    Used for the pure-FE pair model where dense OLS is infeasible. X_fe includes
    the constant in its column span (full IY dummy set), so r2 uses centered y."""
    k = int(X_fe.shape[1])
    n = len(y)
    iter_lim = max(5 * k, 5_000)
    alpha = sp_lsqr(X_fe, y, atol=1e-8, btol=1e-8, iter_lim=iter_lim)[0]
    resid = y - X_fe @ alpha
    ss_res = float(resid @ resid)
    yc     = y - y.mean()
    ss_tot = float(yc @ yc)
    r2     = 1.0 - ss_res / max(ss_tot, 1e-300)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if (n - k) > 0 else float("nan")
    return r2, adj_r2, k


def fit_fwl_clustered(y: np.ndarray, X_reg: np.ndarray, X_fe: csr_matrix,
                       clusters: np.ndarray, n_fe_cols: int):
    """OLS via Frisch-Waugh-Lovell with a sparse FE matrix and clustered SEs (à la 05b).
    Partials X_fe out of y and each regressor via LSQR, then runs dense OLS on the
    residuals; the FWL theorem makes these coefficients (and residuals) identical to the
    full regressors+FE-dummies regression.

    Returns (params, se, tvals, pvals, r2_overall, adj_r2_overall, within_r2, n).
      r2_overall  — 1 − SSR/SST (centered y); comparable to the other columns' R².
      adj_r2_overall — penalised by k_total = n_fe_cols + n_regressors (degenerate when
                       pair FEs saturate the sample, exactly as in Model 4).
      within_r2   — fraction of the FE-partialled-out variation in y explained by the
                    regressors; the standard fit metric for high-dimensional FE models.
    """
    n, k = X_reg.shape
    k_total   = n_fe_cols + k     # regressors + absorbed FE params (dense-column dof convention)
    dof_resid = n - k_total       # residual dof; can be tiny/<=0 when the pair FEs saturate
    iter_lim = max(5 * X_fe.shape[1], 5_000)

    def partial_out(v: np.ndarray) -> np.ndarray:
        alpha = sp_lsqr(X_fe, v, atol=1e-8, btol=1e-8, iter_lim=iter_lim)[0]
        return v - X_fe @ alpha

    My = partial_out(y)
    MX = np.column_stack([partial_out(X_reg[:, j]) for j in range(k)])

    params, _, _, _ = np.linalg.lstsq(MX, My, rcond=None)
    resid = My - MX @ params          # = full-model residuals (FWL theorem)

    # Clustered sandwich SE
    unique_g, g_inv = np.unique(clusters, return_inverse=True)
    G = len(unique_g)
    meat = np.zeros((k, k))
    for gi in range(G):
        mask = g_inv == gi
        score = MX[mask].T @ resid[mask]
        meat += np.outer(score, score)

    MXtMX_inv = np.linalg.inv(MX.T @ MX)
    # A4: count the absorbed FE parameters in the finite-sample dof, exactly as the dense
    # columns do (statsmodels' clustered correction uses K = all columns incl. FE dummies in
    # (n-1)/(n-K)). Using k = regressors only understated K and made these SEs anticonservative
    # vs reghdfe. When the pair FEs saturate the sample there is little/no residual dof, so the
    # correction is large (SEs inflated); with no residual dof it is undefined -> NaN.
    if dof_resid > 0:
        correction = (G / (G - 1)) * ((n - 1) / dof_resid)
    else:
        correction = np.nan
    V  = correction * MXtMX_inv @ meat @ MXtMX_inv
    se = np.sqrt(np.diag(V).clip(0))

    tvals = params / np.where(se > 0, se, np.nan)
    pvals = 2 * t_dist.sf(np.abs(tvals), df=G - 1)

    ss_res  = float(resid @ resid)
    yc      = y - y.mean()
    ss_tot  = float(yc @ yc)
    r2      = 1.0 - ss_res / max(ss_tot, 1e-300)
    adj_r2  = 1.0 - (1.0 - r2) * (n - 1) / dof_resid if dof_resid > 0 else float("nan")

    My_c      = My - My.mean()
    ss_within = float(My_c @ My_c)
    within_r2 = 1.0 - ss_res / max(ss_within, 1e-300)

    return params, se, tvals, pvals, r2, adj_r2, within_r2, n


# ── Design-matrix cleaner ──────────────────────────────────────────────────────

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

    The FE blocks are complete dummy sets fitted with no intercept, so the design is linearly
    dependent by construction: the industry-year dummies sum to 1 on every row, and so do the
    borrower dummies, hence sum(iy) − sum(borrower) = 0.  A rank-deficient matrix this size is
    handed by statsmodels' default `pinv` path to LAPACK's `gesdd` SVD, which INTERMITTENTLY
    FAILS TO CONVERGE — `numpy.linalg.LinAlgError: SVD did not converge`.  It is a numerical
    coin-flip, not a property of any particular column.  This guard makes the solve well-posed
    so a re-run cannot die at random.  Ported from 07/08 so the dense RQ1 columns (1–3, 5, 7)
    use the same rank handling as the RQ2/RQ3 baselines.

    It does NOT change any estimate.  The dropped columns add nothing to the column space, so
    fitted values, residuals, R² and every reported coefficient are identical in exact
    arithmetic.  Columns are ordered [regressors, then FE] and an unpivoted QR drops a column
    only when it is dependent on those already accepted, so the regressors of interest (first,
    and not in the FE span) can never be the ones dropped.  R² stays UNCENTERED
    (`hasconst=False`), so nothing about the reported table changes.
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


# ── Estimation ──────────────────────────────────────────────────────────────────

def fit_ols_clustered(y: pd.Series, X: pd.DataFrame, clusters: pd.Series):
    model = sm.OLS(y, X, hasconst=False)
    return model.fit(cov_type="cluster",
                     cov_kwds={"groups": clusters, "use_correction": True})


# ── Output formatting ────────────────────────────────────────────────────────────

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


def winsorize_cols(frame: pd.DataFrame, cols: list, mask: pd.Series, p: float = 0.01):
    """Return a copy of `frame` with each column in `cols` winsorized at [p, 1-p],
    with bounds computed on the rows selected by `mask` (the regression sample).
    Missing values stay missing (clip does not fill NaN). Prints per-column diagnostics."""
    out = frame.copy()
    for c in cols:
        if c not in out.columns:
            continue
        lo = out.loc[mask, c].quantile(p)
        hi = out.loc[mask, c].quantile(1 - p)
        n_clip = int(((out[c] < lo) | (out[c] > hi)).sum())
        out[c] = out[c].clip(lower=lo, upper=hi)
        print(f"    {c:<20} [{lo:.4f}, {hi:.4f}]  ({n_clip} obs clipped)")
    return out


def model_column(coefs, tvals, pvals, n_obs: int, r2: float, adj_r2: float,
                  fe_counts: dict, fe_label_order: list,
                  master_regressors: list, dv_name: str) -> list:
    """Return values list: DV name, then coef/t-stat rows aligned to master_regressors,
    then footer rows (N, R², Adj. R², one row per FE type in fe_label_order — blank if
    this model doesn't include that FE type). coefs/tvals/pvals are pandas Series (may be
    empty for a pure-FE model with no regressors)."""
    values = [dv_name]
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


# ── Shared inputs ────────────────────────────────────────────────────────────────

def load_lender_lists() -> pd.Series:
    """Build the (borrower_id, tranche_active_date) → [lender_parent_id, …] mapping
    from dealscan raw. DV-independent, so computed once and reused across all DVs."""
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


# ── Per-DV table builder ──────────────────────────────────────────────────────────

def build_table(df_full: pd.DataFrame, lender_lists: pd.Series,
                sheet_name: str, score_cols: list) -> pd.DataFrame:
    """Run all 8 models for one dependent variable and return the formatted table.
    dv = "{sheet_name}_score_dummy" = 1 if ANY of score_cols > 0, else 0."""
    dv = f"{sheet_name}_score_dummy"
    print(f"\n{'#' * 60}\n#  {sheet_name}:  {dv}  "
          f"(1 if any of {score_cols} > 0)\n{'#' * 60}")

    # Sample filters
    df = df_full[df_full["claude_is_debt_contract"] == "Y"].copy()
    print(f"  {len(df):,} rows after keeping claude_is_debt_contract == Y")
    df = df[df[score_cols].notna().all(axis=1)].copy()
    print(f"  {len(df):,} rows after dropping rows missing any of {score_cols}")

    # Dependent variable and regressors
    df[dv]              = (df[score_cols] > 0).any(axis=1).astype(float)
    df["contract_year"] = df["tranche_active_date"].dt.year

    # accounting_policy (formerly no842): OR logic — 1 if EITHER claude_GAAP_OVERRIDE_SCORE
    # or claude_FREEZE_SCORE is positive (non-zero); 0 if both are zero; NaN when both
    # source scores are missing (missing kept as missing, not coerced to 0).
    gaap   = df["claude_GAAP_OVERRIDE_SCORE"]
    freeze = df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan

    print(f"\n{dv} distribution:")
    print(f"  {df[dv].value_counts().sort_index().to_dict()}")
    print(f"accounting_policy distribution (dropna=False):")
    print(f"  {df['accounting_policy'].value_counts(dropna=False).sort_index().to_dict()}")

    # ── Fixed effects ────────────────────────────────────────────────────────
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

    # ── Lender multi-hot FEs (from the prebuilt DV-independent lender_lists) ───
    df = df.join(lender_lists, on=MERGE_KEYS)
    df["lender_ids"] = df["lender_ids"].apply(lambda x: x if isinstance(x, list) else [])
    n_with_lenders = (df["lender_ids"].apply(len) > 0).sum()
    print(f"  {n_with_lenders:,} / {len(df):,} tranches matched to at least one lender")

    fe_lender = make_lender_multi_hot(df, df["lender_ids"])
    print(f"Lender FE matrix: {fe_lender.shape[1]} columns")

    clusters = df["gvkey"]
    y        = df[dv]

    # ── Regression samples + winsorization ────────────────────────────────────
    # Two determinant-based samples, each with its own winsorization on its own rows:
    #   m5: all 5 determinants non-missing (Models 5 & 6)
    #   m7: all 5 determinants AND all controls non-missing (Models 7 & 8)
    # Winsorization (1% both tails) is applied to the non-logged/non-dummy level vars,
    # computed on each model group's own sample. Missing stays missing.
    def _sample_mask(frame, cols):
        finite = np.isfinite(frame[cols].to_numpy(dtype=float)).all(axis=1)
        return (pd.Series(finite, index=frame.index)
                & frame[dv].notna() & frame["gvkey"].notna())

    m5_mask = _sample_mask(df, DETERMINANTS)
    print(f"\nModel 5/6 sample (all 5 determinants non-missing): {int(m5_mask.sum()):,}")
    print("  winsorizing level determinants at 1% on this sample:")
    df5w = winsorize_cols(df, ["offbslease"], m5_mask)   # only offbslease is a level determinant

    df7 = df.copy()
    m7_mask = _sample_mask(df7, DETERMINANTS + CONTROLS)
    print(f"Model 7/8 sample ({len(DETERMINANTS)} determinants + {len(CONTROLS)} controls non-missing): {int(m7_mask.sum()):,}")
    print("  winsorizing level determinants+controls at 1% on this sample:")
    df7w = winsorize_cols(df7, WINSOR_LEVEL_VARS, m7_mask)

    # Prebuilt regressor frames (index-aligned to df; stabilize_design drops missing rows)
    X5_reg = df5w[DETERMINANTS]
    X7_reg = df7w[DETERMINANTS + CONTROLS]

    IY_LABEL   = f"Industry×Year FEs (SIC {sic_digits}-digit)"
    BOR_LABEL  = "Borrower FEs"
    LEN_LABEL  = "Lender FEs"
    PAIR_LABEL = "Borrower-Lender Pair FEs"
    fe_label_order = [IY_LABEL, BOR_LABEL, LEN_LABEL, PAIR_LABEL]   # union across all models

    col_data  = {}
    # Dense models: (prebuilt regressor frame or None, FE blocks).
    #   1–3 FE-only; 5 = determinants on Model 3 FE; 7 = determinants+controls on Model 3 FE.
    # Sparse models 4 (FE-only), 6, 8 (with regressors via FWL) are handled after this loop.
    col_specs = {
        "(1)": (None,   [X_fe]),
        "(2)": (None,   [X_fe, fe_bor]),
        "(3)": (None,   [X_fe, fe_bor, fe_lender]),
        "(5)": (X5_reg, [X_fe, fe_bor, fe_lender]),
        "(7)": (X7_reg, [X_fe, fe_bor, fe_lender]),
    }

    # Master regressor row order = determinants then controls (union across all models).
    master_regressors = DETERMINANTS + CONTROLS

    for col_name, (X_reg_frame, fe_parts) in col_specs.items():
        print(f"\n{'=' * 60}\n  Model {col_name}\n{'=' * 60}")
        X_reg = (X_reg_frame.astype(float) if X_reg_frame is not None
                 else pd.DataFrame(index=df.index))
        X_full = pd.concat([X_reg] + fe_parts, axis=1)
        X, y_clean, cl_clean = stabilize_design(X_full, y, clusters)
        res = fit_ols_clustered(y_clean, X, cl_clean)

        iy_cols  = [c for c in X.columns if c.startswith("iy_")]
        b_cols   = [c for c in X.columns if c.startswith("b_")]
        l_cols   = [c for c in X.columns if c.startswith("l_")]
        # A FE type is reported only if this model actually includes it (columns survived).
        fe_counts = {IY_LABEL: len(iy_cols)}
        if b_cols:
            fe_counts[BOR_LABEL] = len(b_cols)
        if l_cols:
            fe_counts[LEN_LABEL] = len(l_cols)

        col_data[col_name] = model_column(
            res.params, res.tvalues, res.pvalues, len(y_clean),
            res.rsquared, res.rsquared_adj, fe_counts, fe_label_order,
            master_regressors, dv,
        )

        print(f"  N = {len(y_clean):,}  |  R² = {res.rsquared:.4f}  |  Adj. R² = {res.rsquared_adj:.4f}")
        print(f"  IY FEs: {len(iy_cols)}"
              + (f"  |  Borrower FEs: {len(b_cols)}" if b_cols else "")
              + (f"  |  Lender FEs: {len(l_cols)}" if l_cols else ""))
        print(f"  Unique clusters (gvkey): {cl_clean.nunique():,}")

    # ── Model (4): Industry×Year + Borrower-Lender Pair FEs (sparse, FE-only) ──
    # Pair FEs are far too numerous (~13K cols) for dense OLS; build sparse and
    # project y via LSQR (mirrors 05b's FE handling). No regressors, so no
    # coefficients/SEs — we report only N, R², Adj. R², and FE counts.
    print(f"\n{'=' * 60}\n  Model (4): IY + Borrower-Lender Pair FEs (sparse)\n{'=' * 60}")
    y4 = y.to_numpy(dtype=float)

    # IY dense→sparse, drop constants and singletons (sum ≤ 1)
    iy_sp   = csr_matrix(X_fe.to_numpy(dtype=float))
    iy_sums = np.asarray(iy_sp.sum(axis=0)).ravel()
    iy_sp   = iy_sp[:, iy_sums > 1]
    n_iy_pre = int(iy_sp.shape[1])

    # Pair FEs sparse, drop singletons
    fe_pair, _ = make_pair_multi_hot_sparse(df, df["lender_ids"])
    pair_sums  = np.asarray(fe_pair.sum(axis=0)).ravel()
    fe_pair    = fe_pair[:, pair_sums > 1]

    # Combine and dedup across IY + pair (catches cross-type identical columns)
    fe_all = sp_hstack([iy_sp, fe_pair], format="csr")
    fe_all = _dedup_sparse_cols(fe_all)
    n_iy_active   = min(n_iy_pre, int(fe_all.shape[1]))
    n_pair_active = int(fe_all.shape[1]) - n_iy_active

    r2_4, adj_r2_4, k4 = fe_only_r2_sparse(y4, fe_all)
    fe_counts_4 = {IY_LABEL: n_iy_active, PAIR_LABEL: n_pair_active}
    empty = pd.Series(dtype=float)
    col_data["(4)"] = model_column(
        empty, empty, empty, len(y4), r2_4, adj_r2_4,
        fe_counts_4, fe_label_order, master_regressors, dv,
    )
    print(f"  N = {len(y4):,}  |  R² = {r2_4:.4f}  |  Adj. R² = {adj_r2_4:.4f}  (k={k4:,}, n-k={len(y4)-k4:,})")
    print(f"  IY FEs: {n_iy_active}  |  Pair FEs: {n_pair_active}")

    # ── Sparse FWL models (6, 8): Model 4's IY + Borrower-Lender Pair FE structure,
    # with regressors absorbed via Frisch-Waugh-Lovell and clustered SEs. Model 6 adds
    # the 5 determinants (df5w sample); Model 8 adds determinants + controls (df7w sample).
    def run_sparse_fwl(col_name, data_sub, regressors, label):
        print(f"\n{'=' * 60}\n  Model {col_name}: {label}\n{'=' * 60}")
        y_s  = data_sub[dv].to_numpy(dtype=float)
        X_s  = data_sub[regressors].to_numpy(dtype=float)
        cl_s = data_sub["gvkey"].to_numpy()

        iy_d, _  = make_industry_year_fe(data_sub, sic_digits)
        iy_s     = csr_matrix(iy_d.values.astype(float))
        iy_sums  = np.asarray(iy_s.sum(axis=0)).ravel()
        iy_s     = iy_s[:, iy_sums > 1]
        n_iy_pre = int(iy_s.shape[1])

        pr, _  = make_pair_multi_hot_sparse(data_sub, data_sub["lender_ids"])
        pr_sum = np.asarray(pr.sum(axis=0)).ravel()
        pr     = pr[:, pr_sum > 1]

        fe = sp_hstack([iy_s, pr], format="csr")
        fe = _dedup_sparse_cols(fe)
        n_iy   = min(n_iy_pre, int(fe.shape[1]))
        n_pair = int(fe.shape[1]) - n_iy

        # Drop any regressor constant on the subset
        reg_keep = X_s.std(axis=0, ddof=0) > 0
        X_s      = X_s[:, reg_keep]
        names    = [d for d, k in zip(regressors, reg_keep) if k]

        params, se, tv, pv, r2, adj_r2, within, n = fit_fwl_clustered(
            y_s, X_s, fe, cl_s, n_iy + n_pair,
        )
        fe_counts = {IY_LABEL: n_iy, PAIR_LABEL: n_pair}
        col_data[col_name] = model_column(
            pd.Series(params, index=names), pd.Series(tv, index=names),
            pd.Series(pv, index=names), n, r2, adj_r2,
            fe_counts, fe_label_order, master_regressors, dv,
        )
        print(f"  N = {n:,}  |  R² = {r2:.4f}  |  Adj. R² = {adj_r2:.4f}  |  Within R² = {within:.4f}")
        print(f"  IY FEs: {n_iy}  |  Pair FEs: {n_pair}  |  clusters (gvkey): {len(np.unique(cl_s)):,}")

    data6 = df5w.loc[m5_mask].reset_index(drop=True)
    run_sparse_fwl("(6)", data6, DETERMINANTS,
                   "IY + Pair FEs + 5 determinants (FWL)")

    data8 = df7w.loc[m7_mask].reset_index(drop=True)
    run_sparse_fwl("(8)", data8, DETERMINANTS + CONTROLS,
                   "IY + Pair FEs + 5 determinants + controls (FWL)")

    footer_labels = ["", "N", "R²", "Adj. R²"] + fe_label_order
    full_index    = ["Dependent variable"] + _build_labels(master_regressors) + footer_labels
    # Sparse models are computed after the dense loop, so col_data insertion order is
    # (1),(2),(3),(5),(7),(4),(6),(8); reindex to the intended (1)…(8) left-to-right.
    col_order = ["(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)"]
    return pd.DataFrame(col_data, index=full_index)[col_order]


# ── Main ─────────────────────────────────────────────────────────────────────────

def run():
    print("Loading contracts …")
    df_full = pd.read_parquet(DATA_DIR / "fulldata.parquet")
    print(f"  {len(df_full):,} rows × {df_full.shape[1]} cols")

    lender_lists = load_lender_lists()   # DV-independent — built once, reused per DV

    tables = {}
    for sheet_name, score_cols in DV_SPECS:
        tables[sheet_name] = build_table(df_full, lender_lists, sheet_name, score_cols)

    OUT_DIR.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        for sheet_name, _ in DV_SPECS:
            tables[sheet_name].to_excel(xw, sheet_name=sheet_name)

    sheets = ", ".join(name for name, _ in DV_SPECS)
    print(f"\nSaved → {OUT_FILE}  (sheets: {sheets})")


if __name__ == "__main__":
    run()
