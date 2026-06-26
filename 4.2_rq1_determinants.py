"""
4.2_rq1_determinants.py
RQ1: What explains the design of off-balance-sheet lease covenants?

Dependent variables:
  claude_SLB_SCORE  — ordered 0–3 (continuous OLS)
  SLB_score_dummy   — 1 if claude_SLB_SCORE > 0, else 0 (linear probability)

FE structure (FE-B from full_regression_v2.py):
  Industry×Year FEs (2-digit SIC × year) + Borrower FEs + Lender FEs (multi-hot)
  Standard errors clustered by gvkey

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
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR.parent / "data"
OUT_DIR  = REPO_DIR.parent / "output"
OUT_FILE = OUT_DIR / "rq1_determinants_v2.xlsx"

MERGE_KEYS = ["borrower_id", "tranche_active_date"]


# ── Fixed-effect builders ─────────────────────────────────────────────────────

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
    """One column per unique lender_parent_id appearing in any deal in the sample."""
    unique_lenders = sorted({lid for lst in lender_lists for lid in lst})
    if not unique_lenders:
        return pd.DataFrame(index=df.index)
    data = {
        f"l_{int(lid)}": lender_lists.apply(lambda lst, v=lid: 1.0 if v in lst else 0.0)
        for lid in unique_lenders
    }
    return pd.DataFrame(data, index=df.index)


# ── Design-matrix cleaner ─────────────────────────────────────────────────────

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

    print(f"    stabilize: {n_dropped} non-finite rows dropped, {X.shape[1]} cols remaining")
    return X.astype(float), y, clusters


# ── Estimation ────────────────────────────────────────────────────────────────

def fit_ols_clustered(y: pd.Series, X: pd.DataFrame, clusters: pd.Series):
    model = sm.OLS(y, X, hasconst=False)
    return model.fit(cov_type="cluster",
                     cov_kwds={"groups": clusters, "use_correction": True})


# ── Output formatting ─────────────────────────────────────────────────────────

def _stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1:  return "*"
    return ""


# Fixed row order covering all regressors across all models.
MASTER_REGRESSORS = [
    "num_rating",
    "non_rated",
    "claude_GAAP_OVERRIDE_SCORE",
    "claude_FREEZE_SCORE",
    "no842",
]

# Dealscan deal-characteristic controls (all constructed in 2_dealscan.py)
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

# Compustat firm-level controls (all constructed in 1_compustat.py)
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

# Subset of FIRM_CONTROLS that are not already logged and not dummies → winsorize at 1%
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
    """Return values list aligned to master_regressors rows, then footer rows."""
    coefs  = res.params
    tvals  = res.tvalues
    pvals  = res.pvalues

    values = []
    for var in master_regressors:
        if var in coefs.index:
            values.append(f"{coefs[var]:.3f}{_stars(pvals[var])}")
            values.append(f"({tvals[var]:.3f})")
        else:
            values.append("")
            values.append("")

    values.append("")                          # blank separator
    values.append(f"{int(n_obs):,}")           # N
    values.append(f"{res.rsquared_adj:.4f}")   # Adj. R²
    for v in fe_counts.values():
        values.append(str(v))

    return values


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("Loading contracts …")
    df = pd.read_parquet(DATA_DIR / "contracts.parquet")
    print(f"  {len(df):,} rows × {df.shape[1]} cols")

    # Sample filters
    df = df[df["claude_is_debt_contract"] == "Y"].copy()
    print(f"  {len(df):,} rows after keeping claude_is_debt_contract == Y")
    df = df[df["claude_SLB_SCORE"].notna()].copy()
    print(f"  {len(df):,} rows after dropping missing claude_SLB_SCORE")

    # Dependent variables
    df["SLB_score_dummy"] = (df["claude_SLB_SCORE"] > 0).astype(float)
    df["contract_year"]   = df["tranche_active_date"].dt.year

    # no842: 1 if both GAAP_OVERRIDE and FREEZE scores are non-zero
    df["no842"] = (
        (df["claude_GAAP_OVERRIDE_SCORE"].fillna(0) != 0) &
        (df["claude_FREEZE_SCORE"].fillna(0) != 0)
    ).astype(int)

    # is_covenant_ratio is 0 by definition when fin_covenant_count == 0
    df["is_covenant_ratio"] = df["is_covenant_ratio"].fillna(0)

    print(f"\nclause_SLB_SCORE distribution:")
    print(f"  {df['claude_SLB_SCORE'].value_counts().sort_index().to_dict()}")
    print(f"SLB_score_dummy distribution:")
    print(f"  {df['SLB_score_dummy'].value_counts().sort_index().to_dict()}")
    print(f"no842 distribution:")
    print(f"  {df['no842'].value_counts().sort_index().to_dict()}")

    # ── Lender multi-hot FEs (FE-B pattern from full_regression_v2.py) ────────
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
    df["lender_ids"] = df["lender_ids"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    n_with_lenders = (df["lender_ids"].apply(len) > 0).sum()
    print(f"  {n_with_lenders:,} / {len(df):,} tranches matched to at least one lender")

    # ── Fixed effects ─────────────────────────────────────────────────────────
    N          = len(df)
    sic_digits = 2
    fe_iy, n_cells = make_industry_year_fe(df, sic_digits)
    while n_cells >= N - 5 and sic_digits > 1:
        sic_digits -= 1
        fe_iy, n_cells = make_industry_year_fe(df, sic_digits)

    fe_bor    = make_borrower_fe(df)
    fe_lender = make_lender_multi_hot(df, df["lender_ids"])

    X_fe = pd.concat([fe_iy, fe_bor, fe_lender], axis=1).fillna(0.0)
    X_fe = X_fe.loc[:, (X_fe != 0).any(axis=0)]

    print(f"\nFE matrix: {X_fe.shape[1]} columns "
          f"({fe_iy.shape[1]} IY, {fe_bor.shape[1]} Borrower, {fe_lender.shape[1]} Lender)")

    BASE_REGRESSORS       = ["num_rating", "non_rated"]
    COMPONENT_REGRESSORS  = BASE_REGRESSORS + ["claude_GAAP_OVERRIDE_SCORE", "claude_FREEZE_SCORE"]
    NO842_REGRESSORS      = BASE_REGRESSORS + ["no842"]

    def run_specs(spec_regressors: dict, master_regressors: list, label: str,
                  data: pd.DataFrame) -> pd.DataFrame:
        print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
        col_data  = {}
        fe_counts = {}
        for col_name, (y, regressors) in spec_regressors.items():
            print(f"\n── {col_name} ──────────────────────────")
            X_reg  = data[regressors].copy().astype(float)
            X_full = pd.concat([X_reg, X_fe], axis=1)
            X, y_clean, cl_clean = stabilize_design(X_full, y, data["gvkey"])
            res = fit_ols_clustered(y_clean, X, cl_clean)

            iy_cols  = [c for c in X.columns if c.startswith("iy_")]
            b_cols   = [c for c in X.columns if c.startswith("b_")]
            l_cols   = [c for c in X.columns if c.startswith("l_")]
            fe_counts = {
                f"Industry×Year FEs (SIC {sic_digits}-digit)": len(iy_cols),
                "Borrower FEs":                                 len(b_cols),
                "Lender FEs":                                   len(l_cols),
            }
            col_data[col_name] = model_column(res, len(y_clean), fe_counts, master_regressors)

            print(f"  N = {len(y_clean):,}  |  Adj. R² = {res.rsquared_adj:.4f}")
            print(f"  IY FEs: {len(iy_cols)}  |  Borrower FEs: {len(b_cols)}  |  Lender FEs: {len(l_cols)}")
            print(f"  Unique clusters (gvkey): {cl_clean.nunique():,}")

        footer_labels = ["", "N", "Adj. R²"] + list(fe_counts.keys())
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

    # Determine regression sample for winsorization
    all_ac_vars = list(dict.fromkeys(COMP_AC + NO842_AC))
    sample_mask = (
        df[all_ac_vars].apply(lambda s: np.isfinite(s.astype(float))).all(axis=1)
        & np.isfinite(df["claude_SLB_SCORE"].astype(float))
        & np.isfinite(df["SLB_score_dummy"])
        & df["gvkey"].notna()
    )
    print(f"\nAll-controls regression sample: {sample_mask.sum():,} rows "
          f"({len(df) - sample_mask.sum():,} dropped for missing vars)")

    # Winsorize non-logged, non-dummy firm vars at 1% on regression sample
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

    # ── Save output ───────────────────────────────────────────────────────────
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
    X_c   = X_vif - X_vif.mean()   # center to avoid intercept-driven inflation
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
