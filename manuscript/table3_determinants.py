"""
table3_determinants.py  —  Table 3: RQ1 determinants
=====================================================

RQ1, one specification estimated for all five dependent variables side by side, in the
manuscript's Table 3 layout: a single sheet with five DV columns ordered ALL, SLB, SYN,
OPL, VAR-RES.

  Fixed effects   : Industry×Year (2-digit SIC × year) + Borrower + Lender (multi-hot)
  Right-hand side : the determinants + the deal/firm controls (DETERMINANTS, CONTROLS)
  Estimator       : linear probability model, no separate intercept (the complete
                    Industry×Year block already spans the constant)
  Standard errors : clustered by gvkey, finite-sample corrected
  Cells           : coefficient + significance stars only — NO t-statistics

RATING SPEC. Credit quality enters as bucket dummies, not the linear 0–22 scale:
BB_grade / B_grade / CCC_below / non_rated_suppl_all are included and **ig_grade
(investment grade, BBB− or above) is the OMITTED REFERENCE**, so it is not a regressor.
This matches the spec adopted across the pipeline in 7985df5.

R² and Adj. R² are CENTERED (statsmodels reports uncentered under hasconst=False because
the FE span includes the constant). Level variables are winsorized 1%/99% on the
estimation sample. The design matrix and its rank clean are built ONCE and reused across
the five DVs — the row set and columns are DV-invariant, since every DV is finite on the
estimation sample.

Rehash of exploratory/06d_rq1_table3_robust.py, trimmed (dead _build_labels removed,
ROW_ORDER reduced to variable names since only those are displayed, console condensed).
Coefficients are identical to that script's; only the output filename and sheet name differ.

Input : data/contracts.parquet, data/dealscan_raw.parquet
Output: output/tables/table3.xlsx  (single sheet 'Table 3', five DV columns)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_DIR = Path(__file__).resolve().parent.parent   # code/ — this script lives in code/manuscript/
DATA_DIR = REPO_DIR.parent / "data"
OUT_DIR  = REPO_DIR.parent / "output" / "tables"
OUT_FILE = OUT_DIR / "table3.xlsx"
SHEET    = "Table 3"

# Five DVs → five columns; DV = 1 if ANY listed score > 0. Order matches the manuscript.
DV_SPECS = [
    ("ALL",     ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
                 "claude_VAR_SCORE", "claude_RES_SCORE"]),
    ("SLB",     ["claude_SLB_SCORE"]),
    ("SYN",     ["claude_SYN_SCORE"]),
    ("OPL",     ["claude_OPL_SCORE"]),
    ("VAR-RES", ["claude_VAR_SCORE", "claude_RES_SCORE"]),
]
MERGE_KEYS = ["borrower_id", "tranche_active_date"]
DV_SCORES  = ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
              "claude_VAR_SCORE", "claude_RES_SCORE"]

# ig_grade is the OMITTED rating reference — deliberately absent from DETERMINANTS.
DETERMINANTS = ["accounting_policy", "offbslease", "BB_grade", "B_grade", "CCC_below",
                "non_rated_suppl_all", "relationship_freq"]
CONTROLS = [
    "maturity", "log_lender_count", "log_interest", "log_deal_amount", "perf_pricing",
    "fin_covenant_count", "gen_covenant_count", "secured",
    "size", "profitability", "bsfixed", "liabilities", "logage", "btm", "capex",
    "loss", "rand", "divyield",
    "log_bond_count", "bond_proceeds_scaled",
]
WINSOR_LEVEL_VARS = [
    "offbslease",
    "fin_covenant_count", "gen_covenant_count",
    "profitability", "bsfixed", "liabilities", "btm", "capex", "rand", "divyield",
    "bond_proceeds_scaled",
]

# Presentation order for the table rows. Must cover DETERMINANTS + CONTROLS exactly once
# (asserted in run()); the regression itself uses DETERMINANTS + CONTROLS, not this list.
# Rows display the ORIGINAL variable names — manuscript labels live in exploratory/06c/06d.
ROW_ORDER = [
    "offbslease", "BB_grade", "B_grade", "CCC_below", "non_rated_suppl_all",
    "relationship_freq", "secured", "perf_pricing", "fin_covenant_count",
    "accounting_policy", "maturity", "log_lender_count", "log_interest",
    "log_deal_amount", "gen_covenant_count", "size", "profitability", "bsfixed",
    "liabilities", "logage", "btm", "capex", "loss", "rand", "divyield",
    "log_bond_count", "bond_proceeds_scaled",
]


# ── Fixed-effect builders ────────────────────────────────────────────────────────

def make_industry_year_fe(df: pd.DataFrame, sic_digits: int = 2):
    sic_str  = df["sic"].fillna(0).astype(int).astype(str).str.zfill(4).str[:sic_digits]
    ind_year = sic_str + "_" + df["contract_year"].astype(str)
    return pd.get_dummies(ind_year, prefix="iy", dtype=float), int(ind_year.nunique())


def make_borrower_fe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df["borrower_id"].astype("Int64"), prefix="b", dtype=float)


def make_lender_multi_hot(df: pd.DataFrame, lender_lists: pd.Series) -> pd.DataFrame:
    s = lender_lists.explode().dropna()
    if s.empty:
        return pd.DataFrame(index=df.index)
    oh = (pd.get_dummies(s.astype(int), prefix="l", prefix_sep="_", dtype=float)
          .groupby(level=0).max())
    return oh.reindex(df.index, fill_value=0.0)


def load_lender_lists() -> pd.Series:
    ds = pd.read_parquet(DATA_DIR / "dealscan_raw.parquet",
                         columns=MERGE_KEYS + ["lender_parent_id"])
    ds["tranche_active_date"] = pd.to_datetime(ds["tranche_active_date"], errors="coerce")
    ds = ds.dropna(subset=["lender_parent_id"])
    ds["lender_parent_id"] = ds["lender_parent_id"].astype(int)
    return ds.groupby(MERGE_KEYS)["lender_parent_id"].apply(list).rename("lender_ids")


# ── Design cleaner + estimation ──────────────────────────────────────────────────

def drop_dependent_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are exact linear combinations of columns to their LEFT (unpivoted
    QR), so the identified rank is well-defined. Estimate-preserving (span unchanged)."""
    Xv   = X.to_numpy(dtype=float)
    _, R = np.linalg.qr(Xv, mode="reduced")
    diag = np.abs(np.diag(R))
    keep = diag > diag.max() * max(Xv.shape) * np.finfo(float).eps
    if (n_drop := int((~keep).sum())):
        print(f"    rank guard: dropped {n_drop} dependent column(s) "
              f"({X.shape[1]} → {int(keep.sum())})")
    return X.loc[:, keep]


def stabilize_design(X: pd.DataFrame, y: pd.Series, clusters: pd.Series):
    """Drop non-finite/unclustered rows, then constant / duplicate / singleton / dependent
    columns."""
    row_ok = (np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
              & np.isfinite(y.to_numpy(dtype=float))
              & clusters.notna().to_numpy())
    if (n_dropped := int((~row_ok).sum())):
        print(f"    stabilize: {n_dropped} non-finite row(s) dropped")
    X, y, clusters = X.loc[row_ok], y.loc[row_ok], clusters.loc[row_ok]

    X = X.loc[:, X.std(ddof=0) > 0]      # constant
    X = X.loc[:, ~X.T.duplicated()]      # duplicate
    X = X.loc[:, X.sum(axis=0) != 1]     # singleton
    return drop_dependent_columns(X).astype(float), y, clusters


def fit_ols_clustered(y: pd.Series, X: pd.DataFrame, clusters: pd.Series):
    return sm.OLS(y, X, hasconst=False).fit(
        cov_type="cluster", cov_kwds={"groups": clusters, "use_correction": True})


def centered_r2(res, n_params: int):
    """CENTERED R²/Adj. R² for the no-intercept FE model. R²_c = 1 − SSR/Σ(y−ȳ)²;
    Adj = 1 − (1−R²_c)(n−1)/(n−k). Coefficients and SEs are unaffected."""
    n  = int(res.nobs)
    r2 = 1.0 - res.ssr / res.centered_tss
    return r2, (1.0 - (1.0 - r2) * (n - 1) / (n - n_params) if n > n_params else float("nan"))


def winsorize_cols(frame: pd.DataFrame, cols: list, mask: pd.Series, p: float = 0.01):
    out = frame.copy()
    for c in cols:
        if c not in out.columns:
            continue
        lo, hi = out.loc[mask, c].quantile(p), out.loc[mask, c].quantile(1 - p)
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out


def _stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


# ── Main ─────────────────────────────────────────────────────────────────────────

def run() -> None:
    assert set(ROW_ORDER) == set(DETERMINANTS + CONTROLS) and len(ROW_ORDER) == len(set(ROW_ORDER)), (
        "ROW_ORDER must cover DETERMINANTS + CONTROLS exactly once; symmetric diff: "
        f"{set(DETERMINANTS + CONTROLS) ^ set(ROW_ORDER)}")

    print("Loading contracts …")
    df = pd.read_parquet(DATA_DIR / "contracts.parquet")
    print(f"  {len(df):,} rows × {df.shape[1]} cols")

    df = df[df["claude_is_debt_contract"] == "Y"]
    df = df[df[DV_SCORES].notna().all(axis=1)].copy()
    print(f"  {len(df):,} rows after debt-contract + non-missing-score filters")

    df["contract_year"] = df["tranche_active_date"].dt.year
    gaap, freeze = df["claude_GAAP_OVERRIDE_SCORE"], df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan
    for sheet, cols in DV_SPECS:
        df[f"{sheet}_score_dummy"] = (df[cols] > 0).any(axis=1).astype(float)

    # Industry×Year, backing off SIC granularity if the cells would exhaust the sample.
    N, sic_digits = len(df), 2
    fe_iy, n_cells = make_industry_year_fe(df, sic_digits)
    while n_cells >= N - 5 and sic_digits > 1:
        sic_digits -= 1
        fe_iy, n_cells = make_industry_year_fe(df, sic_digits)
    X_fe   = fe_iy.loc[:, (fe_iy != 0).any(axis=0)]
    fe_bor = make_borrower_fe(df)

    df = df.join(load_lender_lists(), on=MERGE_KEYS)
    df["lender_ids"] = df["lender_ids"].apply(lambda x: x if isinstance(x, list) else [])
    fe_lender = make_lender_multi_hot(df, df["lender_ids"])

    finite  = np.isfinite(df[DETERMINANTS + CONTROLS].to_numpy(dtype=float)).all(axis=1)
    m7_mask = pd.Series(finite, index=df.index) & df["gvkey"].notna()
    print(f"  estimation sample ({len(DETERMINANTS)} determinants + {len(CONTROLS)} controls "
          f"non-missing): {int(m7_mask.sum()):,}")
    X_reg = winsorize_cols(df, WINSOR_LEVEL_VARS, m7_mask)[DETERMINANTS + CONTROLS]

    FE_BLOCKS = [(f"Industry×Year FEs (SIC {sic_digits}-digit)", "iy_"),
                 ("Borrower FEs", "b_"),
                 ("Lender FEs",   "l_")]

    # Design + rank clean done ONCE — DV-invariant, since every DV is finite on this sample.
    X_full  = pd.concat([X_reg.astype(float), X_fe, fe_bor, fe_lender], axis=1)
    y_probe = df[f"{DV_SPECS[-1][0]}_score_dummy"]
    X, _, clusters = stabilize_design(X_full, y_probe, df["gvkey"])
    fe_counts = {lab: sum(c.startswith(pre) for c in X.columns) for lab, pre in FE_BLOCKS}
    print(f"  FEs — " + "  ".join(f"{lab.split(' FE')[0]}={fe_counts[lab]}" for lab, _ in FE_BLOCKS)
          + f"  |  clusters (gvkey): {clusters.nunique():,}")

    col_data = {}
    for sheet, _ in DV_SPECS:
        dv  = f"{sheet}_score_dummy"
        y   = df.loc[X.index, dv]
        res = fit_ols_clustered(y, X, clusters)
        r2c, adjc = centered_r2(res, X.shape[1])
        print(f"  {sheet:<8} N={len(y):,}  R²={r2c:.4f}  Adj.R²={adjc:.4f}  (centered)")

        col_data[sheet] = (
            [dv]
            + [f"{res.params[v]:.3f}{_stars(res.pvalues[v])}" if v in res.params.index else ""
               for v in ROW_ORDER]
            + ["", f"{len(y):,}", f"{r2c:.4f}", f"{adjc:.4f}"]
            + [str(fe_counts[lab]) for lab, _ in FE_BLOCKS])

    index = (["Dependent variable"] + ROW_ORDER + ["", "N", "R²", "Adj. R²"]
             + [lab for lab, _ in FE_BLOCKS])
    table = pd.DataFrame(col_data, index=index)[[s for s, _ in DV_SPECS]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        table.to_excel(xw, sheet_name=SHEET)
        ws = xw.sheets[SHEET]
        ws.set_column(0, 0, 30)
        ws.set_column(1, table.shape[1], 16)
    print(f"\nSaved → {OUT_FILE}  (sheet: {SHEET}; columns: {', '.join(s for s, _ in DV_SPECS)})")


if __name__ == "__main__":
    run()
