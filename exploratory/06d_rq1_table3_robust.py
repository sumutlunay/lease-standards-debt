"""
06d_rq1_table3_robust.py  →  output/tables/table3_robust.xlsx   (Table 3, robustness)

RQ1 Table 3 with the BUCKET-RATING robustness specification: the linear credit-rating variable
(num_rating_suppl_all) is replaced by credit-quality bucket dummies. Otherwise identical to 06c
(same DVs, FE structure, controls, sample, centered R², manuscript layout).

Rating spec: ig_grade (investment grade, BBB- or above) is the OMITTED reference category; the
included rating dummies are BB_grade, B_grade, CCC_below and non_rated_suppl_all (built in 03 on
num_rating_suppl_all). accounting_policy and offbslease are unchanged.

Industry×Year + Borrower + Lender multi-hot FEs, with the determinants AND the deal/firm controls,
reported as ONE sheet with the five DVs in five columns ordered ALL, SLB, SYN, OPL, VAR-RES.

Presentation (manuscript Table 3):
  • DV column order: ALL, SLB, SYN, OPL, VAR-RES.
  • Rows are the regressors in the manuscript's order with the manuscript's display labels
    (see ROW_ORDER) — NOT the DETERMINANTS+CONTROLS code order.
  • Each cell is the coefficient with significance stars only; t-statistics are NOT reported.
  • Footer keeps N, R² (centered), Adj. R² (centered) and the FE counts.

  Fixed effects   : Industry×Year (2-digit SIC × year) + Borrower + Lender (multi-hot on
                    lender_parent_id). Dense OLS, SEs clustered by gvkey, with the shared
                    singleton/duplicate/linear-dependence rank guard (as in 06/07/08).
  Right-hand side : the determinants (incl. the rating buckets) + the deal/firm controls
                    (see DETERMINANTS, CONTROLS); ig_grade omitted as the rating reference.
  Sample          : m7 — a row enters iff all determinants AND all controls are non-missing
                    (bond_proceeds_scaled is NaN where Compustat total assets ≤ 0, so it also
                    trims the sample). Identical across the five DV columns → N = 11,184.
  Winsorization   : non-logged, non-dummy level variables at 1% both tails on the m7 sample.
  R²              : reported CENTERED (about the DV mean; see centered_r2) — coefficients/SEs
                    are unaffected by the centering.

Because the five DVs share one sample and one FE structure, the design matrix is built and
rank-cleaned ONCE and only the dependent vector changes across the five fits.

Input:  data/contracts.parquet       (output of 03_contracts.py + 04_merge.py)
        data/dealscan_raw.parquet    (lender_parent_id multi-hot FEs)
Output: output/tables/table3_robust.xlsx    (single sheet 'Table 3 robust', five DV columns)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_DIR = Path(__file__).resolve().parent.parent  # code/ — this script lives in code/exploratory/
DATA_DIR = REPO_DIR.parent / "data"
OUT_DIR  = REPO_DIR.parent / "output" / "tables"
OUT_FILE = OUT_DIR / "table3_robust.xlsx"
SHEET    = "Table 3 robust"

# Five DVs → five columns (short name, [raw claude score columns]); DV = 1 if ANY score > 0.
# Column order matches the manuscript's Table 3: ALL first, then SLB, SYN, OPL, VAR-RES.
DV_SPECS = [
    ("ALL",     ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
                 "claude_VAR_SCORE", "claude_RES_SCORE"]),
    ("SLB",     ["claude_SLB_SCORE"]),
    ("SYN",     ["claude_SYN_SCORE"]),
    ("OPL",     ["claude_OPL_SCORE"]),
    ("VAR-RES", ["claude_VAR_SCORE", "claude_RES_SCORE"]),
]
MERGE_KEYS = ["borrower_id", "tranche_active_date"]

# Bucket-rating robustness spec: the linear num_rating_suppl_all is REPLACED by credit-quality
# bucket dummies (built in 03). ig_grade (investment grade) is the OMITTED reference category,
# so it is NOT a regressor; BB_grade / B_grade / CCC_below / non_rated_suppl_all are included.
DETERMINANTS = ["accounting_policy", "offbslease", "BB_grade", "B_grade", "CCC_below", "non_rated_suppl_all", "relationship_freq"]
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

# Output row order + display labels for the manuscript's Table 3 (code variable → manuscript
# label). This is the PRESENTATION order only; the regression uses DETERMINANTS + CONTROLS.
# Every entry must be a regressor in the model, and every model regressor must appear here
# exactly once (asserted in run()).
ROW_ORDER = [
    ("offbslease",           "Operating lease intensity"),
    ("BB_grade",             "BB"),
    ("B_grade",              "B"),
    ("CCC_below",            "CCC and below"),
    ("non_rated_suppl_all",  "Unrated"),
    ("relationship_freq",    "Relationship lender"),
    ("secured",              "Secured"),
    ("perf_pricing",         "Performance pricing"),
    ("fin_covenant_count",   "Financial covenant"),
    ("accounting_policy",    "Accounting policy"),
    ("maturity",             "Maturity"),
    ("log_lender_count",     "Number of lenders"),
    ("log_interest",         "Loan spread"),
    ("log_deal_amount",      "Loan amount"),
    ("gen_covenant_count",   "General covenant"),
    ("size",                 "Firm size"),
    ("profitability",        "Profitability"),
    ("bsfixed",              "Tangibility"),
    ("liabilities",          "Leverage"),
    ("logage",               "Firm age"),
    ("btm",                  "Book-to-market"),
    ("capex",                "Capex"),
    ("loss",                 "Loss"),
    ("rand",                 "R&D"),
    ("divyield",             "Dividend yield"),
    ("log_bond_count",       "Public bond issuance"),
    ("bond_proceeds_scaled", "Public bond proceeds"),
]


# ── Fixed-effect builders (identical to 06) ──────────────────────────────────────

def make_industry_year_fe(df: pd.DataFrame, sic_digits: int = 2):
    sic_str  = df["sic"].fillna(0).astype(int).astype(str).str.zfill(4).str[:sic_digits]
    ind_year = sic_str + "_" + df["contract_year"].astype(str)
    fe       = pd.get_dummies(ind_year, prefix="iy", drop_first=False, dtype=float)
    return fe, int(ind_year.nunique())


def make_borrower_fe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df["borrower_id"].astype("Int64"), prefix="b",
                          drop_first=False, dtype=float)


def make_lender_multi_hot(df: pd.DataFrame, lender_lists: pd.Series) -> pd.DataFrame:
    s = lender_lists.explode().dropna()
    if s.empty:
        return pd.DataFrame(index=df.index)
    s  = s.astype(int)
    oh = pd.get_dummies(s, prefix="l", prefix_sep="_", dtype=float).groupby(level=0).max()
    return oh.reindex(df.index, fill_value=0.0)


def load_lender_lists() -> pd.Series:
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


# ── Design cleaner + estimation (identical to 06) ────────────────────────────────

def drop_dependent_columns(X: pd.DataFrame) -> pd.DataFrame:
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


def stabilize_design(X: pd.DataFrame, y: pd.Series, clusters: pd.Series):
    row_ok = (
        np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
        & np.isfinite(y.to_numpy(dtype=float))
        & clusters.notna().to_numpy()
    )
    n_dropped = int((~row_ok).sum())
    X, y, clusters = X.loc[row_ok], y.loc[row_ok], clusters.loc[row_ok]
    X = X.loc[:, X.std(ddof=0) > 0]
    X = X.loc[:, ~X.T.duplicated()]
    X = X.loc[:, X.sum(axis=0) != 1]
    X = drop_dependent_columns(X)
    print(f"    stabilize: {n_dropped} non-finite rows dropped, {X.shape[1]} cols remaining")
    return X.astype(float), y, clusters


def fit_ols_clustered(y: pd.Series, X: pd.DataFrame, clusters: pd.Series):
    model = sm.OLS(y, X, hasconst=False)
    return model.fit(cov_type="cluster",
                     cov_kwds={"groups": clusters, "use_correction": True})


def centered_r2(res, n_params: int):
    """CENTERED R² / Adj. R² for the no-intercept FE model (statsmodels reports UNCENTERED under
    hasconst=False because the FE span includes the constant). R²_c = 1 − SSR/Σ(y−ȳ)²;
    Adj = 1 − (1−R²_c)(n−1)/(n−k). Coefficients/SEs unaffected. See 06's centered_r2."""
    n  = int(res.nobs)
    r2 = 1.0 - res.ssr / res.centered_tss
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - n_params) if n > n_params else float("nan")
    return r2, adj


def winsorize_cols(frame: pd.DataFrame, cols: list, mask: pd.Series, p: float = 0.01):
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


# ── Main ─────────────────────────────────────────────────────────────────────────

def run():
    print("Loading contracts …")
    df = pd.read_parquet(DATA_DIR / "contracts.parquet")
    print(f"  {len(df):,} rows × {df.shape[1]} cols")

    # Sample filters (identical to 06/07/08)
    df = df[df["claude_is_debt_contract"] == "Y"].copy()
    df = df[df[["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
                "claude_VAR_SCORE", "claude_RES_SCORE"]].notna().all(axis=1)].copy()
    print(f"  {len(df):,} rows after debt-contract + non-missing-score filters")

    # Constructed regressors + the five DV dummies
    df["contract_year"] = df["tranche_active_date"].dt.year
    gaap   = df["claude_GAAP_OVERRIDE_SCORE"]
    freeze = df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan
    for sheet, cols in DV_SPECS:
        df[f"{sheet}_score_dummy"] = (df[cols] > 0).any(axis=1).astype(float)

    # ── Fixed effects: IY + Borrower + Lender (multi-hot) ─────────────────────
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

    lender_lists = load_lender_lists()
    df = df.join(lender_lists, on=MERGE_KEYS)
    df["lender_ids"] = df["lender_ids"].apply(lambda x: x if isinstance(x, list) else [])
    fe_lender = make_lender_multi_hot(df, df["lender_ids"])
    print(f"Lender FE matrix: {fe_lender.shape[1]} columns")

    # ── m7 sample + winsorization (shared across the five DVs) ────────────────
    finite  = np.isfinite(df[DETERMINANTS + CONTROLS].to_numpy(dtype=float)).all(axis=1)
    m7_mask = pd.Series(finite, index=df.index) & df["gvkey"].notna()
    print(f"\nEstimation sample ({len(DETERMINANTS)} determinants + {len(CONTROLS)} controls "
          f"non-missing): {int(m7_mask.sum()):,}")
    print("  winsorizing level variables at 1% on this sample:")
    dfw   = winsorize_cols(df, WINSOR_LEVEL_VARS, m7_mask)
    X_reg = dfw[DETERMINANTS + CONTROLS]

    IY_LABEL  = f"Industry×Year FEs (SIC {sic_digits}-digit)"
    BOR_LABEL = "Borrower FEs"
    LEN_LABEL = "Lender FEs"
    fe_label_order = [IY_LABEL, BOR_LABEL, LEN_LABEL]

    # Design matrix + rank clean done ONCE — DV-invariant (row set and columns do not depend
    # on which DV is regressed, since every DV is finite on the m7 sample).
    X_full = pd.concat([X_reg.astype(float), X_fe, fe_bor, fe_lender], axis=1)
    y_probe = df[f"{DV_SPECS[-1][0]}_score_dummy"]           # any DV → same row_ok
    X, _, clusters = stabilize_design(X_full, y_probe, df["gvkey"])
    fe_counts = {
        IY_LABEL:  sum(c.startswith("iy_") for c in X.columns),
        BOR_LABEL: sum(c.startswith("b_")  for c in X.columns),
        LEN_LABEL: sum(c.startswith("l_")  for c in X.columns),
    }
    print(f"  IY: {fe_counts[IY_LABEL]}  |  Borrower: {fe_counts[BOR_LABEL]}  "
          f"|  Lender: {fe_counts[LEN_LABEL]}  |  clusters (gvkey): {clusters.nunique():,}")

    master_regressors = DETERMINANTS + CONTROLS
    # ROW_ORDER must present each model regressor exactly once (guards a rename/typo/omission).
    assert {v for v, _ in ROW_ORDER} == set(master_regressors), (
        "ROW_ORDER must cover exactly DETERMINANTS + CONTROLS; symmetric diff: "
        f"{set(master_regressors) ^ {v for v, _ in ROW_ORDER}}"
    )

    col_data = {}
    for sheet, _ in DV_SPECS:
        dv = f"{sheet}_score_dummy"
        y  = df.loc[X.index, dv]
        res = fit_ols_clustered(y, X, clusters)
        r2c, adjc = centered_r2(res, X.shape[1])   # CENTERED (see centered_r2)
        print(f"\n  {sheet:<8} DV={dv}  N={len(y):,}  R²={r2c:.4f}  Adj.R²={adjc:.4f}  (centered)")

        # One cell per regressor in the manuscript's ROW_ORDER: coefficient + significance
        # stars only — NO t-statistics.
        values = [dv]
        for var, _label in ROW_ORDER:
            values.append(f"{res.params[var]:.3f}{_stars(res.pvalues[var])}"
                          if var in res.params.index else "")
        values += ["", f"{len(y):,}", f"{r2c:.4f}", f"{adjc:.4f}"]
        values += [str(fe_counts[l]) for l in fe_label_order]
        col_data[sheet] = values

    index = (["Dependent variable"] + [var for var, _ in ROW_ORDER]   # ORIGINAL variable names
             + ["", "N", "R²", "Adj. R²"] + fe_label_order)
    table = pd.DataFrame(col_data, index=index)[[s for s, _ in DV_SPECS]]

    OUT_DIR.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        table.to_excel(xw, sheet_name=SHEET)
        ws = xw.sheets[SHEET]
        ws.set_column(0, 0, 30)
        ws.set_column(1, table.shape[1], 16)

    print(f"\nSaved → {OUT_FILE}  (sheet: {SHEET}; columns: {', '.join(s for s, _ in DV_SPECS)})")


if __name__ == "__main__":
    run()
