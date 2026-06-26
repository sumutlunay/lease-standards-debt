"""
4.2_rq1_determinants.py
RQ1: What explains the design of off-balance-sheet lease covenants?

Dependent variables:
  claude_SLB_SCORE  — ordered 0–3 (continuous OLS)
  SLB_score_dummy   — 1 if claude_SLB_SCORE > 0, else 0 (linear probability)

FE structure (FE-D from full_regression_v2.py):
  Industry×Year FEs (2-digit SIC × year) + Borrower-Lender Pair FEs (multi-hot)
  Standard errors clustered by gvkey

Estimation: Frisch-Waugh-Lovell (FWL) with sparse pair FE matrix.
  Pair FEs are too numerous for dense SVD (~46K columns); instead:
    1. Build pair FE as scipy.sparse.csr_matrix (vectorized, memory-efficient)
    2. Partial FEs out of y and each regressor via LSQR
    3. Run dense OLS on the small partialled-out regressor matrix
    4. Compute clustered SEs via the sandwich estimator on the partialled-out data

Three sets of specifications:
  no_controls   — base regressors + FEs only
  deal_controls — base regressors + Dealscan deal characteristics + FEs
  all_controls  — deal_controls + Compustat firm characteristics + FEs
                  non-logged, non-dummy firm vars winsorized at 1% on regression sample

VIF diagnostics are printed for the all_controls / Components spec at the end.

Input:  data/contracts.parquet       (output of 3_contracts_merge.py)
        data/dealscan_raw.parquet    (for lender_parent_id multi-hot FEs)
Output: output/rq1_determinants_v2.xlsx  (sheets: no_controls, deal_controls, all_controls)
"""

from pathlib import Path
import types
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sp_hstack
from scipy.sparse.linalg import lsqr as sp_lsqr
from scipy.stats import t as t_dist
from statsmodels.stats.outliers_influence import variance_inflation_factor

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR.parent / "data"
OUT_DIR  = REPO_DIR.parent / "output"
OUT_FILE = OUT_DIR / "rq1_determinants_v2.xlsx"

MERGE_KEYS = ["borrower_id", "tranche_active_date"]


# ── Fixed-effect builders ─────────────────────────────────────────────────────

def make_industry_year_fe(df: pd.DataFrame, sic_digits: int = 2):
    """Dense dummies for (SIC-{sic_digits}d industry) × year."""
    sic_str  = df["sic"].fillna(0).astype(int).astype(str).str.zfill(4).str[:sic_digits]
    ind_year = sic_str + "_" + df["contract_year"].astype(str)
    fe       = pd.get_dummies(ind_year, prefix="iy", drop_first=False, dtype=float)
    return fe, int(ind_year.nunique())


def make_pair_multi_hot_sparse(df: pd.DataFrame, lender_lists: pd.Series):
    """
    Sparse (borrower_id × lender_parent_id) pair FEs.
    Returns (csr_matrix [n_rows × n_pairs], col_names list).
    Vectorized: builds index arrays then constructs the sparse matrix in one call.
    """
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
    """
    Remove exact-duplicate columns from a sparse matrix.
    Two columns are duplicates if they have identical non-zero row indices.
    Uses byte-hashing of each column's index array — O(nnz) total.
    """
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


# ── FWL estimator with sparse FEs ────────────────────────────────────────────

def fit_fwl_clustered(y: np.ndarray, X_reg: np.ndarray, X_fe,
                       clusters: np.ndarray, n_fe_cols: int):
    """
    OLS via Frisch-Waugh-Lovell with a sparse FE matrix and clustered SEs.

    Partials X_fe out of y and each column of X_reg via LSQR, then runs
    dense OLS on the residuals.  The FWL theorem guarantees that these
    coefficients equal those from the full (regressors + FE dummy) regression.

    Returns (params, se, tvalues, pvalues, rsquared_adj, n).
    """
    n, k = X_reg.shape
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

    MXtMX     = MX.T @ MX
    MXtMX_inv = np.linalg.inv(MXtMX)
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    V  = correction * MXtMX_inv @ meat @ MXtMX_inv
    se = np.sqrt(np.diag(V).clip(0))

    tvals = params / np.where(se > 0, se, np.nan)
    pvals = 2 * t_dist.sf(np.abs(tvals), df=G - 1)

    # Within R²: fraction of FE-partialled-out variation explained by X_reg.
    # Overall adj. R² breaks down when n_pair_FEs ≈ n (df denominator near zero).
    # Within R² is the standard metric for high-dimensional FE models.
    ss_res    = float(resid @ resid)
    My_c      = My - My.mean()
    ss_within = float(My_c @ My_c)
    within_r2 = 1.0 - ss_res / max(ss_within, 1e-300)

    return params, se, tvals, pvals, within_r2, n


# ── Output formatting ─────────────────────────────────────────────────────────

def _stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1:  return "*"
    return ""


MASTER_REGRESSORS = [
    "num_rating",
    "non_rated",
    "claude_GAAP_OVERRIDE_SCORE",
    "claude_FREEZE_SCORE",
    "no842",
]

DEAL_CONTROLS = [
    "maturity",
    "log_lender_count",
    "log_interest",
    "log_deal_amount",
    "perf_pricing",
    "fin_covenant_count",
    "gen_covenant_count",
    "is_covenant_ratio",
    "secured",
]

FIRM_CONTROLS = [
    "size",
    "profitability",
    "bsfixed",
    "liabilities",
    "offbslease",
    "logage",
    "btm",
    "capex",
    "loss",
    "rand",
    "divyield",
]

FIRM_CONTROLS_WINSORIZE = [
    "profitability",
    "bsfixed",
    "liabilities",
    "offbslease",
    "btm",
    "capex",
    "rand",
    "divyield",
]


def _build_labels(regressors: list) -> list:
    labels = []
    for v in regressors:
        labels += [v, ""]
    return labels


def model_column(res, n_obs: int, fe_counts: dict, master_regressors: list) -> list:
    coefs = res.params
    tvals = res.tvalues
    pvals = res.pvalues

    values = []
    for var in master_regressors:
        if var in coefs.index:
            values.append(f"{coefs[var]:.3f}{_stars(pvals[var])}")
            values.append(f"({tvals[var]:.3f})")
        else:
            values.append("")
            values.append("")

    values.append("")
    values.append(f"{int(n_obs):,}")
    values.append(f"{res.rsquared_adj:.4f}")
    for v in fe_counts.values():
        values.append(str(v))

    return values


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("Loading contracts …")
    df = pd.read_parquet(DATA_DIR / "contracts.parquet")
    print(f"  {len(df):,} rows × {df.shape[1]} cols")

    df = df[df["claude_is_debt_contract"] == "Y"].copy()
    print(f"  {len(df):,} rows after keeping claude_is_debt_contract == Y")
    df = df[df["claude_SLB_SCORE"].notna()].copy()
    print(f"  {len(df):,} rows after dropping missing claude_SLB_SCORE")

    df["SLB_score_dummy"] = (df["claude_SLB_SCORE"] > 0).astype(float)
    df["contract_year"]   = df["tranche_active_date"].dt.year

    df["no842"] = (
        (df["claude_GAAP_OVERRIDE_SCORE"].fillna(0) != 0) &
        (df["claude_FREEZE_SCORE"].fillna(0) != 0)
    ).astype(int)

    df["is_covenant_ratio"] = df["is_covenant_ratio"].fillna(0)

    print(f"\nclause_SLB_SCORE distribution:")
    print(f"  {df['claude_SLB_SCORE'].value_counts().sort_index().to_dict()}")
    print(f"SLB_score_dummy distribution:")
    print(f"  {df['SLB_score_dummy'].value_counts().sort_index().to_dict()}")
    print(f"no842 distribution:")
    print(f"  {df['no842'].value_counts().sort_index().to_dict()}")

    # ── Lender IDs ────────────────────────────────────────────────────────────
    print("\nLoading lender parent IDs from dealscan raw …")
    ds_raw = pd.read_parquet(
        DATA_DIR / "dealscan_raw.parquet",
        columns=["borrower_id", "tranche_active_date", "lender_parent_id"],
    )
    ds_raw["tranche_active_date"] = pd.to_datetime(ds_raw["tranche_active_date"], errors="coerce")
    ds_raw = ds_raw.dropna(subset=["lender_parent_id"])
    ds_raw["lender_parent_id"] = ds_raw["lender_parent_id"].astype(int)

    lender_lists = (
        ds_raw.groupby(MERGE_KEYS)["lender_parent_id"]
        .apply(list)
        .rename("lender_ids")
    )
    df = df.join(lender_lists, on=MERGE_KEYS)
    df["lender_ids"] = df["lender_ids"].apply(lambda x: x if isinstance(x, list) else [])
    n_with_lenders = (df["lender_ids"].apply(len) > 0).sum()
    print(f"  {n_with_lenders:,} / {len(df):,} tranches matched to at least one lender")

    # ── Determine SIC digits from full sample ─────────────────────────────────
    N = len(df)
    sic_digits = 2
    _, n_cells = make_industry_year_fe(df, sic_digits)
    while n_cells >= N - 5 and sic_digits > 1:
        sic_digits -= 1
        _, n_cells = make_industry_year_fe(df, sic_digits)
    print(f"\nUsing {sic_digits}-digit SIC ({n_cells} IY cells in full sample)")

    BASE_REGRESSORS      = ["num_rating", "non_rated"]
    COMPONENT_REGRESSORS = BASE_REGRESSORS + ["claude_GAAP_OVERRIDE_SCORE", "claude_FREEZE_SCORE"]
    NO842_REGRESSORS     = BASE_REGRESSORS + ["no842"]

    # ── Spec runner ───────────────────────────────────────────────────────────

    def run_specs(spec_regressors: dict, master_regressors: list, label: str,
                  data: pd.DataFrame) -> pd.DataFrame:
        print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
        col_data       = {}
        fe_counts_last = {}

        for col_name, (y, regressors) in spec_regressors.items():
            print(f"\n── {col_name} ──────────────────────────")

            # 1. Filter rows
            X_reg  = data[regressors].copy().astype(float)
            y_s    = y.astype(float)
            cl_s   = data["gvkey"]
            row_ok = (
                np.isfinite(X_reg.to_numpy()).all(axis=1)
                & np.isfinite(y_s.to_numpy())
                & cl_s.notna().to_numpy()
            )
            n_dropped = int((~row_ok).sum())

            data_sub = data.loc[row_ok].copy().reset_index(drop=True)
            X_sub    = X_reg.loc[row_ok].reset_index(drop=True)
            y_sub    = y_s.loc[row_ok].reset_index(drop=True)
            cl_sub   = cl_s.loc[row_ok].reset_index(drop=True)

            # 2. Build FEs on this subset
            fe_iy_sub, _  = make_industry_year_fe(data_sub, sic_digits)
            fe_pair_sub, _ = make_pair_multi_hot_sparse(data_sub, data_sub["lender_ids"])

            # 3. Clean FE columns — mirrors dense stabilize_design: drop constants,
            #    singletons, and duplicates, with cross-type dedup (IY vs pair).
            iy_sp = csr_matrix(fe_iy_sub.values.astype(float))

            # Remove constant IY cols (all-zero) and singletons (sum == 1)
            iy_sums = np.asarray(iy_sp.sum(axis=0)).ravel()
            iy_sp   = iy_sp[:, (iy_sums > 1)]

            # Remove singleton pair cols
            pair_sums   = np.asarray(fe_pair_sub.sum(axis=0)).ravel()
            fe_pair_sub = fe_pair_sub[:, pair_sums > 1]

            # Combined dedup across IY + pair (catches cross-type identical columns)
            n_iy_pre  = int(iy_sp.shape[1])
            fe_all    = sp_hstack([iy_sp, fe_pair_sub], format="csr")
            fe_all    = _dedup_sparse_cols(fe_all)

            n_iy_active   = min(n_iy_pre, int(fe_all.shape[1]))
            n_pair_active = int(fe_all.shape[1]) - n_iy_active

            # 4. Drop constant regressors
            X_arr  = X_sub.values.astype(float)
            X_cols = list(X_sub.columns)
            reg_keep = X_arr.std(axis=0, ddof=0) > 0
            X_arr    = X_arr[:, reg_keep]
            X_cols   = [c for c, k in zip(X_cols, reg_keep) if k]

            # 5. FE matrix is already combined and deduped
            X_fe_comb = fe_all

            print(f"    {n_dropped} rows dropped | "
                  f"{n_iy_active} IY + {n_pair_active} pair FEs | "
                  f"{X_arr.shape[1]} regressors")

            # 6. FWL estimation with clustered SEs
            params, se, tvals, pvals, r2_adj, n = fit_fwl_clustered(
                y_sub.values, X_arr, X_fe_comb, cl_sub.values,
                n_iy_active + n_pair_active,
            )

            result = types.SimpleNamespace(
                params       = pd.Series(params, index=X_cols),
                tvalues      = pd.Series(tvals,  index=X_cols),
                pvalues      = pd.Series(pvals,  index=X_cols),
                rsquared_adj = r2_adj,
            )

            fe_counts_last = {
                f"Industry×Year FEs (SIC {sic_digits}-digit)": n_iy_active,
                "Borrower-Lender Pair FEs":                    n_pair_active,
            }
            col_data[col_name] = model_column(result, n, fe_counts_last, master_regressors)

            print(f"  N = {n:,}  |  Within R² = {r2_adj:.4f}")
            print(f"  IY FEs: {n_iy_active}  |  Pair FEs: {n_pair_active}")
            print(f"  Unique clusters (gvkey): {cl_sub.nunique():,}")
            for chk in ["num_rating", "non_rated"]:
                if chk in result.tvalues.index:
                    tv = result.tvalues[chk]
                    tag = f"t={tv:.3f}" if not np.isnan(tv) else "t=NaN  ← absorbed by pair FEs"
                    print(f"    {chk}: coef={result.params[chk]:.4f}  {tag}")

        footer_labels = ["", "N", "Within R²"] + list(fe_counts_last.keys())
        full_index    = _build_labels(master_regressors) + footer_labels
        return pd.DataFrame(col_data, index=full_index)

    # ── Sheet 1: no_controls ──────────────────────────────────────────────────
    dvs_base = {
        "(1) Continuous / Components": (df["claude_SLB_SCORE"].astype(float), COMPONENT_REGRESSORS),
        "(2) Continuous / no842":      (df["claude_SLB_SCORE"].astype(float), NO842_REGRESSORS),
        "(3) Dummy / Components":      (df["SLB_score_dummy"],                COMPONENT_REGRESSORS),
        "(4) Dummy / no842":           (df["SLB_score_dummy"],                NO842_REGRESSORS),
    }
    tbl_no_controls = run_specs(dvs_base, MASTER_REGRESSORS, "no_controls", df)

    # ── Sheet 2: deal_controls ────────────────────────────────────────────────
    COMP_DC  = COMPONENT_REGRESSORS + DEAL_CONTROLS
    NO842_DC = NO842_REGRESSORS     + DEAL_CONTROLS
    dvs_dc = {
        "(1) Continuous / Components": (df["claude_SLB_SCORE"].astype(float), COMP_DC),
        "(2) Continuous / no842":      (df["claude_SLB_SCORE"].astype(float), NO842_DC),
        "(3) Dummy / Components":      (df["SLB_score_dummy"],                COMP_DC),
        "(4) Dummy / no842":           (df["SLB_score_dummy"],                NO842_DC),
    }
    tbl_deal_controls = run_specs(dvs_dc, MASTER_REGRESSORS + DEAL_CONTROLS, "deal_controls", df)

    # ── Sheet 3: all_controls ─────────────────────────────────────────────────
    COMP_AC  = COMPONENT_REGRESSORS + DEAL_CONTROLS + FIRM_CONTROLS
    NO842_AC = NO842_REGRESSORS     + DEAL_CONTROLS + FIRM_CONTROLS

    all_ac_vars = list(dict.fromkeys(COMP_AC + NO842_AC))
    sample_mask = (
        df[all_ac_vars].apply(lambda s: np.isfinite(s.astype(float))).all(axis=1)
        & np.isfinite(df["claude_SLB_SCORE"].astype(float))
        & np.isfinite(df["SLB_score_dummy"])
        & df["gvkey"].notna()
    )
    print(f"\nAll-controls regression sample: {sample_mask.sum():,} rows "
          f"({len(df) - sample_mask.sum():,} dropped for missing vars)")

    df_ac = df.copy()
    print("Winsorizing firm controls at 1% on regression sample:")
    for col in FIRM_CONTROLS_WINSORIZE:
        lo = df_ac.loc[sample_mask, col].quantile(0.01)
        hi = df_ac.loc[sample_mask, col].quantile(0.99)
        n_clipped = int(((df_ac[col] < lo) | (df_ac[col] > hi)).sum())
        df_ac[col] = df_ac[col].clip(lower=lo, upper=hi)
        print(f"  {col:<15}  [{lo:.4f}, {hi:.4f}]  ({n_clipped} obs clipped)")

    dvs_ac = {
        "(1) Continuous / Components": (df_ac["claude_SLB_SCORE"].astype(float), COMP_AC),
        "(2) Continuous / no842":      (df_ac["claude_SLB_SCORE"].astype(float), NO842_AC),
        "(3) Dummy / Components":      (df_ac["SLB_score_dummy"],                COMP_AC),
        "(4) Dummy / no842":           (df_ac["SLB_score_dummy"],                NO842_AC),
    }
    tbl_all_controls = run_specs(dvs_ac, MASTER_REGRESSORS + DEAL_CONTROLS + FIRM_CONTROLS,
                                 "all_controls", df_ac)

    OUT_DIR.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        tbl_no_controls.to_excel(xw, sheet_name="no_controls")
        tbl_deal_controls.to_excel(xw, sheet_name="deal_controls")
        tbl_all_controls.to_excel(xw, sheet_name="all_controls")

    print(f"\nSaved → {OUT_FILE}  (sheets: no_controls, deal_controls, all_controls)")

    # ── VIF check: all_controls / Components spec ─────────────────────────────
    print(f"\n{'=' * 60}")
    print("  VIF diagnostics — all_controls (Components spec)")
    print(f"{'=' * 60}")
    sample = df_ac[sample_mask].copy()
    X_vif = sample[COMP_AC].astype(float).dropna()
    X_c   = X_vif - X_vif.mean()
    vifs  = pd.Series(
        [variance_inflation_factor(X_c.values, i) for i in range(X_c.shape[1])],
        index=X_c.columns,
    ).sort_values(ascending=False)
    print(vifs.round(1).to_string())

    print(f"\n{'=' * 60}")
    print("  VIF diagnostics — all_controls (no842 spec)")
    print(f"{'=' * 60}")
    X_vif2 = sample[NO842_AC].astype(float).dropna()
    X_c2   = X_vif2 - X_vif2.mean()
    vifs2  = pd.Series(
        [variance_inflation_factor(X_c2.values, i) for i in range(X_c2.shape[1])],
        index=X_c2.columns,
    ).sort_values(ascending=False)
    print(vifs2.round(1).to_string())


if __name__ == "__main__":
    run()
