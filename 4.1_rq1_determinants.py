"""
4_rq1_determinants.py
RQ1: What explains the design of off-balance-sheet lease covenants?

Dependent variables:
  claude_SLB_SCORE  — ordered 0–3 (continuous OLS)
  SLB_score_dummy   — 1 if claude_SLB_SCORE > 0, else 0 (linear probability)

Model: OLS with Industry×Year FEs (2-digit SIC × year) + Borrower FEs
       Standard errors clustered by borrower_id

Three sets of specifications:
  no_controls   — base regressors + FEs only
  deal_controls — base regressors + Dealscan deal characteristics + FEs
  all_controls  — deal_controls + Compustat firm characteristics + FEs
                  non-logged, non-dummy firm vars winsorized at 1% on regression sample

Input:  data/contracts.parquet  (output of 3_contracts_merge.py)
Output: output/rq1_determinants.xlsx  (sheets: no_controls, deal_controls, all_controls)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR.parent / "data"
OUT_DIR  = REPO_DIR.parent / "output"
OUT_FILE = OUT_DIR / "rq1_determinants_v1.xlsx"

FE_PREFIXES = ("iy_", "b_")


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
# Columns fill with "" for variables absent from that model.
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

    # no842: 1 if both GAAP_OVERRIDE and FREEZE scores are non-zero; missing treated as 0
    df["no842"] = (
        (df["claude_GAAP_OVERRIDE_SCORE"].fillna(0) != 0) &
        (df["claude_FREEZE_SCORE"].fillna(0) != 0)
    ).astype(int)

    print(f"\nclause_SLB_SCORE distribution:")
    print(f"  {df['claude_SLB_SCORE'].value_counts().sort_index().to_dict()}")
    print(f"SLB_score_dummy distribution:")
    print(f"  {df['SLB_score_dummy'].value_counts().sort_index().to_dict()}")
    print(f"no842 distribution:")
    print(f"  {df['no842'].value_counts().sort_index().to_dict()}")

    # Industry×Year FEs — start at 2-digit SIC, coarsen if too many cells
    N          = len(df)
    sic_digits = 2
    fe_iy, n_cells = make_industry_year_fe(df, sic_digits)
    while n_cells >= N - 5 and sic_digits > 1:
        sic_digits -= 1
        fe_iy, n_cells = make_industry_year_fe(df, sic_digits)

    fe_bor = make_borrower_fe(df)
    X_fe   = pd.concat([fe_iy, fe_bor], axis=1).fillna(0.0)
    X_fe   = X_fe.loc[:, (X_fe != 0).any(axis=0)]

    BASE_REGRESSORS       = ["num_rating", "non_rated"]
    COMPONENT_REGRESSORS  = BASE_REGRESSORS + ["claude_GAAP_OVERRIDE_SCORE", "claude_FREEZE_SCORE"]
    NO842_REGRESSORS      = BASE_REGRESSORS + ["no842"]

    # is_covenant_ratio is 0 by definition when fin_covenant_count == 0
    df["is_covenant_ratio"] = df["is_covenant_ratio"].fillna(0)

    clusters = df["gvkey"]

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

            iy_cols = [c for c in X.columns if c.startswith("iy_")]
            b_cols  = [c for c in X.columns if c.startswith("b_")]
            fe_counts = {
                f"Industry×Year FEs (SIC {sic_digits}-digit)": len(iy_cols),
                "Borrower FEs":                                 len(b_cols),
            }
            col_data[col_name] = model_column(res, len(y_clean), fe_counts, master_regressors)

            print(f"  N = {len(y_clean):,}  |  Adj. R² = {res.rsquared_adj:.4f}")
            print(f"  Industry×Year FEs: {len(iy_cols)}  |  Borrower FEs: {len(b_cols)}")
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

    # Determine regression sample: rows finite across all regressors and both DVs
    all_ac_vars = list(dict.fromkeys(COMP_AC + NO842_AC))  # ordered dedup
    sample_mask = (
        df[all_ac_vars].apply(lambda s: np.isfinite(s.astype(float))).all(axis=1)
        & np.isfinite(df["claude_SLB_SCORE"].astype(float))
        & np.isfinite(df["SLB_score_dummy"])
        & df["gvkey"].notna()
    )
    print(f"\nAll-controls regression sample: {sample_mask.sum():,} rows "
          f"({len(df) - sample_mask.sum():,} dropped for missing vars)")

    # Winsorize non-logged, non-dummy firm vars at 1% computed on regression sample
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


if __name__ == "__main__":
    run()
