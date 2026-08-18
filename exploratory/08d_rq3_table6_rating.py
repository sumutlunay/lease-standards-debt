"""
08d_rq3_table6_rating.py — Table 6 with BUCKET-RATING credit quality, ±5-year window only.

A variant of 08b_rq3_table6.py that swaps the LINEAR credit-rating variable for the credit-quality
BUCKET dummies already used in the Table 3 robustness spec (06d_rq1_table3_robust.py):

    num_rating_suppl_all  +  non_rated_suppl_all
        →  BB_grade + B_grade + CCC_below + non_rated_suppl_all
           (ig_grade = investment grade, BBB− or above, is the OMITTED reference category)

All four dummies enter the model twice: as levels and interacted with `post_adoption`, so the
post-adoption shift in the OBSLI–credit-quality relation is estimated non-parametrically relative
to investment-grade borrowers (both pre and post).

Everything else is identical to 08b: adopting firms only, the five DV columns, the Model 7 FE
structure (Industry×Year + Borrower + Lender multi-hot), the same determinants and controls,
`amendment`, `post_adoption` and its interactions, 1% winsorization formed BEFORE the interactions,
dense OLS with SEs clustered by gvkey, and the same rank guard.

★ Only the ±5-year window is estimated here (08b's '10 yr window' sheet), per the specification
request — the ±3-year window is not reproduced.

    adoption_date − 5 years  ≤  tranche_active_date  ≤  adoption_date + 5 years

Input:  data/contracts.parquet, data/dealscan_raw.parquet
Output: output/tables/table6_rating.xlsx   (single sheet '10 yr window' = ±5y)
Format: coefficients + significance stars only (NO t-statistics); R²/Adj. R² reported CENTERED
        (about the DV mean; see centered_r2), matching tables 2–6. SEs clustered by gvkey.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_DIR  = Path(__file__).resolve().parent.parent  # code/ — this script lives in code/exploratory/
DATA_DIR  = REPO_DIR.parent / "data"
OUT_DIR   = REPO_DIR.parent / "output" / "tables"
OUT_FILE  = OUT_DIR / "table6_rating.xlsx"
CONTRACTS = DATA_DIR / "contracts.parquet"

MERGE_KEYS = ["borrower_id", "tranche_active_date"]

# ── The window restriction — ±5 years only in this variant ───────────────────────
WINDOWS = [(5, "10 yr window")]   # (± years, sheet name)

# ── Model 7 regressor lists ──────────────────────────────────────────────────────
DV_SCORES = ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
             "claude_VAR_SCORE", "claude_RES_SCORE"]
DV_SPECS = [
    ("SLB",     ["claude_SLB_SCORE"]),
    ("SYN",     ["claude_SYN_SCORE"]),
    ("OPL",     ["claude_OPL_SCORE"]),
    ("VAR-RES", ["claude_VAR_SCORE", "claude_RES_SCORE"]),
    ("ALL",     DV_SCORES),
]

# Bucket-rating spec (as in 06d): ig_grade is the omitted reference, so it is NOT a regressor.
RATING_BUCKETS = ["BB_grade", "B_grade", "CCC_below", "non_rated_suppl_all"]
DETERMINANTS = ["accounting_policy", "offbslease"] + RATING_BUCKETS + ["relationship_freq"]
CONTROLS = [
    "maturity", "log_lender_count", "log_interest", "log_deal_amount", "perf_pricing",
    "fin_covenant_count", "gen_covenant_count", "secured",
    "size", "profitability", "bsfixed", "liabilities", "logage", "btm", "capex",
    "loss", "rand", "divyield",
    "log_bond_count", "bond_proceeds_scaled",
    "amendment",
]
WINSOR_LEVEL_VARS = [
    "offbslease",
    "fin_covenant_count", "gen_covenant_count",
    "profitability", "bsfixed", "liabilities", "btm", "capex", "rand", "divyield",
    "bond_proceeds_scaled",
]

# ── RQ3A: post_adoption and its interactions ─────────────────────────────────────
# Same interaction set as 08b, except the single linear rating interaction is replaced by one
# interaction per rating bucket (investment grade remains the reference in both regimes).
POST = "post_adoption"
INTERACT_VARS = [
    "accounting_policy",
    "relationship_freq",
    "fin_covenant_count",
    "offbslease",
] + RATING_BUCKETS + [
    "amendment",
]


def interaction_name(var: str) -> str:
    return f"{POST}_x_{var}"


INTERACTIONS = [interaction_name(v) for v in INTERACT_VARS]
TEST_VARS    = [POST] + INTERACTIONS

# The amendment measure used in this table (amendment_seq-based here; 08c uses amendment_claude).
AMENDMENT_VAR = "amendment"

# Manuscript row order + display labels, following 08b with the rating rows replaced by the four
# buckets (levels and interactions). ROW_ORDER must cover exactly the model regressors
# (TEST_VARS + DETERMINANTS + CONTROLS) — asserted in build_rq3a.
ROW_ORDER = [
    ("post_adoption",                          "Post"),
    (interaction_name("accounting_policy"),    "Post×Accounting policy"),
    ("accounting_policy",                      "Accounting policy"),
    (AMENDMENT_VAR,                            "Amendment"),
    (interaction_name(AMENDMENT_VAR),          "Post×Amendment"),
    (interaction_name("relationship_freq"),    "Post×Relationship lender"),
    (interaction_name("fin_covenant_count"),   "Post×Financial covenant"),
    (interaction_name("offbslease"),           "Post×Operating lease intensity"),
    (interaction_name("BB_grade"),             "Post×BB"),
    (interaction_name("B_grade"),              "Post×B"),
    (interaction_name("CCC_below"),            "Post×CCC and below"),
    (interaction_name("non_rated_suppl_all"),  "Post×Unrated"),
    ("relationship_freq",                      "Relationship lender"),
    ("fin_covenant_count",                     "Financial covenant"),
    ("offbslease",                             "Operating lease intensity"),
    ("BB_grade",                               "BB"),
    ("B_grade",                                "B"),
    ("CCC_below",                              "CCC and below"),
    ("non_rated_suppl_all",                    "Unrated"),
    ("maturity",                               "Maturity"),
    # remaining controls (order immaterial):
    ("log_lender_count",     "Number of lenders"),
    ("log_interest",         "Loan spread"),
    ("log_deal_amount",      "Loan amount"),
    ("perf_pricing",         "Performance pricing"),
    ("gen_covenant_count",   "General covenant"),
    ("secured",              "Secured"),
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


# ── Fixed-effect builders (dense; copied from 08b) ───────────────────────────────

def make_industry_year_fe(df: pd.DataFrame, sic_digits: int = 2):
    sic_str  = df["sic"].fillna(0).astype(int).astype(str).str.zfill(4).str[:sic_digits]
    ind_year = sic_str + "_" + df["contract_year"].astype(str)
    fe       = pd.get_dummies(ind_year, prefix="iy", drop_first=False, dtype=float)
    return fe, int(ind_year.nunique())


def make_borrower_fe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df["borrower_id"].astype("Int64"), prefix="b",
                          drop_first=False, dtype=float)


def make_lender_multi_hot(df: pd.DataFrame, lender_lists: pd.Series) -> pd.DataFrame:
    unique_lenders = sorted({lid for lst in lender_lists for lid in lst})
    if not unique_lenders:
        return pd.DataFrame(index=df.index)
    data = {
        f"l_{int(lid)}": lender_lists.apply(lambda lst, v=lid: 1.0 if v in lst else 0.0)
        for lid in unique_lenders
    }
    return pd.DataFrame(data, index=df.index)


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


# ── Estimation helpers (copied from 08b) ─────────────────────────────────────────

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


def _stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1:  return "*"
    return ""


def winsorize_cols(frame: pd.DataFrame, cols: list, mask: pd.Series, p: float = 0.01):
    out = frame.copy()
    for c in cols:
        if c not in out.columns:
            continue
        # Cast to float first: some level vars (e.g. covenant counts) are nullable Int64, and
        # clipping them to a fractional quantile bound raises on Int64. These regressors are cast
        # to float downstream anyway, so this is estimate-preserving.
        out[c] = out[c].astype(float)
        lo = out.loc[mask, c].quantile(p)
        hi = out.loc[mask, c].quantile(1 - p)
        n_clip = int(((out[c] < lo) | (out[c] > hi)).sum())
        out[c] = out[c].clip(lower=lo, upper=hi)
        print(f"    {c:<20} [{lo:.4f}, {hi:.4f}]  ({n_clip} obs clipped)")
    return out


# ── Loan-level sample: adopting firms, restricted to the ±5-year window ──────────

def prepare_sample(lender_lists: pd.Series, window_years: int):
    """Same as 08b's prepare_sample (08's, plus the ±window_years symmetric window filter)."""
    df = pd.read_parquet(CONTRACTS)
    print(f"\nLoaded {CONTRACTS.name}: {len(df):,} rows × {df.shape[1]} cols")

    df = df[df["claude_is_debt_contract"] == "Y"].copy()
    df = df[df[DV_SCORES].notna().all(axis=1)].copy()
    print(f"  {len(df):,} rows after sample filters")

    for sheet, cols in DV_SPECS:
        df[f"{sheet}_score_dummy"] = (df[cols] > 0).any(axis=1).astype(float)

    df["contract_year"] = df["tranche_active_date"].dt.year

    gaap   = df["claude_GAAP_OVERRIDE_SCORE"]
    freeze = df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan

    # post_adoption
    adopt = pd.to_datetime(df["adoption_date"], errors="coerce")
    df[POST] = np.where(adopt.isna(), np.nan,
                        (df["tranche_active_date"] >= adopt).astype(float))

    # Restrict to adopting firms (adoption_date present)
    df = df[df[POST].notna()].copy()
    n_adopt = len(df)
    print(f"\n  Restricted to adopting firms: {n_adopt:,} rows")

    # ── ±window_years symmetric window around adoption_date ─────────────────────
    adopt = pd.to_datetime(df["adoption_date"], errors="coerce")
    lo = adopt - pd.DateOffset(years=window_years)
    hi = adopt + pd.DateOffset(years=window_years)
    in_window = (df["tranche_active_date"] >= lo) & (df["tranche_active_date"] <= hi)
    df = df[in_window].copy()
    print(f"  ±{window_years}-year window around adoption_date: {len(df):,} rows "
          f"({n_adopt - len(df):,} dropped, {len(df)/n_adopt*100:.1f}% kept)")
    print(f"    post_adoption = 1 (post-842) : {int((df[POST] == 1).sum()):,}")
    print(f"    post_adoption = 0 (pre-842)  : {int((df[POST] == 0).sum()):,}")

    for sheet, _ in DV_SPECS:
        d = df[f"{sheet}_score_dummy"]
        print(f"    {sheet + '_score_dummy':<22} mean = {d.mean():.3f}")

    # Rating-bucket composition (ig_grade is the omitted reference — shown for context only).
    print("  rating buckets (ig_grade = omitted reference):")
    for c in ["ig_grade"] + RATING_BUCKETS:
        if c in df.columns:
            print(f"    {c:<22} n = {int(df[c].fillna(0).sum()):,}  "
                  f"({df[c].fillna(0).mean() * 100:.1f}%)")

    # ── Fixed effects (built on the windowed sample) ─────────────────────────────
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
    fe_lender = make_lender_multi_hot(df, df["lender_ids"])
    print(f"Lender FE matrix: {fe_lender.shape[1]} columns")

    fe_labels = [
        f"Industry×Year FEs (SIC {sic_digits}-digit)",
        "Borrower FEs",
        "Lender FEs",
    ]
    return df, X_fe, fe_bor, fe_lender, fe_labels


# ── RQ3A estimation (copied from 08b) ────────────────────────────────────────────

def run_rq3a_column(df, X_fe, fe_bor, fe_lender, fe_labels, sheet: str, col_name: str) -> list:
    dv         = f"{sheet}_score_dummy"
    regressors = TEST_VARS + DETERMINANTS + CONTROLS
    IY_LABEL, BOR_LABEL, LEN_LABEL = fe_labels

    print(f"\n{'=' * 60}\n  RQ3A {col_name}: DV = {dv}\n{'=' * 60}")

    df = df.copy()

    parents = [POST] + DETERMINANTS + CONTROLS
    finite  = np.isfinite(df[parents].to_numpy(dtype=float)).all(axis=1)
    mask    = (pd.Series(finite, index=df.index)
               & df[dv].notna() & df["gvkey"].notna())
    print(f"  Estimation sample (all regressors non-missing): {int(mask.sum()):,}")

    print("  winsorizing level variables at 1% on this sample:")
    dfw = winsorize_cols(df, WINSOR_LEVEL_VARS, mask)
    for v in INTERACT_VARS:
        dfw[interaction_name(v)] = dfw[POST] * dfw[v]

    X_full = pd.concat([dfw[regressors].astype(float), X_fe, fe_bor, fe_lender], axis=1)
    X, y_clean, cl_clean = stabilize_design(X_full, dfw[dv], dfw["gvkey"])
    res = fit_ols_clustered(y_clean, X, cl_clean)
    r2c, adjc = centered_r2(res, X.shape[1])   # CENTERED (see centered_r2)

    iy_cols = [c for c in X.columns if c.startswith("iy_")]
    b_cols  = [c for c in X.columns if c.startswith("b_")]
    l_cols  = [c for c in X.columns if c.startswith("l_")]
    fe_counts = {IY_LABEL: len(iy_cols), BOR_LABEL: len(b_cols), LEN_LABEL: len(l_cols)}

    print(f"  N = {len(y_clean):,}  |  R² = {r2c:.4f}  |  Adj. R² = {adjc:.4f}  (centered)")
    print(f"  IY FEs: {len(iy_cols)}  |  Borrower FEs: {len(b_cols)}  |  Lender FEs: {len(l_cols)}")
    print(f"  Unique clusters (gvkey): {cl_clean.nunique():,}")

    dropped = [v for v in TEST_VARS if v not in X.columns]
    if dropped:
        print(f"  ⚠ dropped by stabilize (collinear/constant): {', '.join(dropped)}")
    for v in TEST_VARS:
        if v in res.params.index:
            print(f"    {v:<34} {res.params[v]:>8.4f}{_stars(res.pvalues[v]):<3} "
                  f"(t={res.tvalues[v]:.3f})")

    # One cell per regressor in the manuscript ROW_ORDER: coefficient + stars only, NO t-stats.
    values = [dv]
    for var, _label in ROW_ORDER:
        values.append(f"{res.params[var]:.3f}{_stars(res.pvalues[var])}"
                      if var in res.params.index else "")

    values.append("")
    values.append(f"{len(y_clean):,}")
    values.append(f"{r2c:.4f}")
    values.append(f"{adjc:.4f}")
    for label in fe_labels:
        values.append(str(fe_counts.get(label, "")))

    return values


def build_rq3a(df, X_fe, fe_bor, fe_lender, fe_labels) -> pd.DataFrame:
    col_data = {}
    for i, (sheet, _) in enumerate(DV_SPECS, start=1):
        col_name = f"({i})"
        col_data[col_name] = run_rq3a_column(
            df, X_fe, fe_bor, fe_lender, fe_labels, sheet, col_name,
        )

    assert {v for v, _ in ROW_ORDER} == set(TEST_VARS + DETERMINANTS + CONTROLS), (
        "ROW_ORDER must cover exactly TEST_VARS + DETERMINANTS + CONTROLS; symmetric diff: "
        f"{set(TEST_VARS + DETERMINANTS + CONTROLS) ^ {v for v, _ in ROW_ORDER}}"
    )
    index = (["Dependent variable"]
             + [var.replace(f"{POST}_x_", f"{POST} × ") for var, _ in ROW_ORDER]   # ORIGINAL variable names
             + ["", "N", "R²", "Adj. R²"] + fe_labels)
    return pd.DataFrame(col_data, index=index)


def run() -> None:
    print(f"\n{'#' * 60}\n#  Table 6 (rating buckets) — RQ3 Eq.(3), ±5y window only "
          f"(centered R², coefficients + stars, no t-stats)\n{'#' * 60}")
    lender_lists = load_lender_lists()
    sheets = {}
    for yrs, sheet_name in WINDOWS:
        print(f"\n{'#' * 60}\n#  ±{yrs}-year window → sheet '{sheet_name}'\n{'#' * 60}")
        df, X_fe, fe_bor, fe_lender, fe_labels = prepare_sample(lender_lists, yrs)
        sheets[sheet_name] = build_rq3a(df, X_fe, fe_bor, fe_lender, fe_labels)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        for name, tab in sheets.items():
            tab.to_excel(xw, sheet_name=name)
            ws = xw.sheets[name]
            ws.set_column(0, 0, 30)
            ws.set_column(1, tab.shape[1], 16)

    print(f"\nSaved → {OUT_FILE}  (sheet: {', '.join(sheets)})")


if __name__ == "__main__":
    run()
