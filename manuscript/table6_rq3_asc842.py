"""
table6_rq3_asc842.py  —  Table 6: RQ3, ASC 842 adoption
========================================================

RQ3 Eq. (3) on symmetric windows around each firm's ASC 842 adoption date, estimated on
ADOPTING FIRMS ONLY. Four panels, one sheet each:

    Panel A   ±3-year window   accounting_policy (composite)
    Panel B   ±5-year window   accounting_policy (composite)
    Panel C   ±3-year window   accounting_policy DECOMPOSED into gaap_override + freeze
    Panel D   ±5-year window   accounting_policy DECOMPOSED into gaap_override + freeze

Panels C/D replace the single `accounting_policy` dummy — and its `post_adoption ×`
interaction — with its two components, each entering as BOTH a main effect and a
`post_adoption ×` interaction, so the moderation can be attributed to the clause type.

  Fixed effects   : Industry×Year + Borrower + Lender (multi-hot), built on the WINDOWED sample
  Standard errors : clustered by gvkey; R²/Adj. R² CENTERED
  Cells           : coefficient + significance stars only — NO t-statistics
  Rating spec     : credit-quality BUCKETS (BB_grade / B_grade / CCC_below /
                    non_rated_suppl_all, ig_grade the omitted reference), each also
                    interacted with post_adoption — as in exploratory/08d

⚠ INTERPRETING PANELS C/D. `freeze = 1` is very nearly a SUBSET of `gaap_override = 1`
(across the full scored sample only 1 contract of 14,584 has freeze without gaap_override),
so `accounting_policy` ≈ `gaap_override`. The two components are correlated ~0.63 and are not
collinear, but the `freeze` coefficient reads as the INCREMENTAL effect for contracts that
also freeze the standard, not as a standalone freeze effect. Watch the console for any
test variable dropped by the rank guard.

Level variables are winsorized 1% BEFORE the interactions are formed, so post × offbslease
and post × fin_covenant_count inherit the winsorized parent.

Non-adopters carry a missing adoption_date and are DROPPED, not coerced to zero — a zero
would pool never-adopters into the pre-period control group. Fixed effects are built after
that restriction so the reported FE counts match the estimated sample.

Rehash of exploratory/08d_rq3_table6_rating.py (bucket ratings) with 08b's two-window
structure, plus the two decomposition panels. Panel B reproduces 08d exactly.

Input : data/contracts.parquet, data/dealscan_raw.parquet
Output: output/tables/table6.xlsx  (sheets: Panel A, Panel B, Panel C, Panel D)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_DIR  = Path(__file__).resolve().parent.parent   # code/ — this script lives in code/manuscript/
DATA_DIR  = REPO_DIR.parent / "data"
OUT_DIR   = REPO_DIR.parent / "output" / "tables"
OUT_FILE  = OUT_DIR / "table6.xlsx"
CONTRACTS = DATA_DIR / "contracts.parquet"
MERGE_KEYS = ["borrower_id", "tranche_active_date"]

# (sheet name, ± years, accounting-policy variables)
COMPOSITE  = ["accounting_policy"]
DECOMPOSED = ["gaap_override", "freeze"]
PANELS = [
    ("Panel A", 3, COMPOSITE),
    ("Panel B", 5, COMPOSITE),
    ("Panel C", 3, DECOMPOSED),
    ("Panel D", 5, DECOMPOSED),
]

DV_SCORES = ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
             "claude_VAR_SCORE", "claude_RES_SCORE"]
DV_SPECS = [
    ("SLB",     ["claude_SLB_SCORE"]),
    ("SYN",     ["claude_SYN_SCORE"]),
    ("OPL",     ["claude_OPL_SCORE"]),
    ("VAR-RES", ["claude_VAR_SCORE", "claude_RES_SCORE"]),
    ("ALL",     DV_SCORES),
]

RATING_BUCKETS = ["BB_grade", "B_grade", "CCC_below", "non_rated_suppl_all"]
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
POST          = "post_adoption"
AMENDMENT_VAR = "amendment"


def interaction_name(var: str) -> str:
    return f"{POST}_x_{var}"


def make_spec(acct_vars: list) -> dict:
    """Regressor lists + row order for one accounting-policy parameterisation.

    Rows display the ORIGINAL variable names (interactions rendered 'post_adoption × x');
    manuscript labels live in exploratory/08b/08d.
    """
    determinants  = acct_vars + ["offbslease"] + RATING_BUCKETS + ["relationship_freq"]
    interact_vars = (acct_vars + ["relationship_freq", "fin_covenant_count", "offbslease"]
                     + RATING_BUCKETS + [AMENDMENT_VAR])
    test_vars     = [POST] + [interaction_name(v) for v in interact_vars]
    # Head = the manuscript's presentation block (test variables, then the levels they
    # interact with). The remaining controls follow in CONTROLS order; filtering against the
    # head keeps fin_covenant_count and amendment from appearing twice.
    head = (
        [POST]
        + [interaction_name(v) for v in acct_vars] + acct_vars
        + [AMENDMENT_VAR, interaction_name(AMENDMENT_VAR)]
        + [interaction_name(v) for v in ["relationship_freq", "fin_covenant_count", "offbslease"]]
        + [interaction_name(v) for v in RATING_BUCKETS]
        + ["relationship_freq", "fin_covenant_count", "offbslease"] + RATING_BUCKETS
    )
    row_order = head + [c for c in CONTROLS if c not in head]
    spec = {"acct": acct_vars, "determinants": determinants, "interact_vars": interact_vars,
            "test_vars": test_vars, "row_order": row_order,
            "regressors": test_vars + determinants + CONTROLS}
    assert set(row_order) == set(spec["regressors"]), (
        f"ROW_ORDER must cover the regressors exactly; symmetric diff: "
        f"{set(spec['regressors']) ^ set(row_order)}")
    assert len(row_order) == len(set(row_order)), "duplicate entry in ROW_ORDER"
    return spec


# ── Fixed-effect builders ────────────────────────────────────────────────────────

def make_industry_year_fe(df: pd.DataFrame, sic_digits: int = 2):
    sic_str  = df["sic"].fillna(0).astype(int).astype(str).str.zfill(4).str[:sic_digits]
    ind_year = sic_str + "_" + df["contract_year"].astype(str)
    return pd.get_dummies(ind_year, prefix="iy", dtype=float), int(ind_year.nunique())


def make_borrower_fe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df["borrower_id"].astype("Int64"), prefix="b", dtype=float)


def make_lender_multi_hot(df: pd.DataFrame, lender_lists: pd.Series) -> pd.DataFrame:
    unique_lenders = sorted({lid for lst in lender_lists for lid in lst})
    if not unique_lenders:
        return pd.DataFrame(index=df.index)
    data = {f"l_{int(lid)}": lender_lists.apply(lambda lst, v=lid: 1.0 if v in lst else 0.0)
            for lid in unique_lenders}
    return pd.DataFrame(data, index=df.index)


def load_lender_lists() -> pd.Series:
    ds = pd.read_parquet(DATA_DIR / "dealscan_raw.parquet",
                         columns=MERGE_KEYS + ["lender_parent_id"])
    ds["tranche_active_date"] = pd.to_datetime(ds["tranche_active_date"], errors="coerce")
    ds = ds.dropna(subset=["lender_parent_id"])
    ds["lender_parent_id"] = ds["lender_parent_id"].astype(int)
    return ds.groupby(MERGE_KEYS)["lender_parent_id"].apply(list).rename("lender_ids")


# ── Design cleaner + estimation ──────────────────────────────────────────────────

def drop_dependent_columns(X: pd.DataFrame) -> pd.DataFrame:
    Xv   = X.to_numpy(dtype=float)
    _, R = np.linalg.qr(Xv, mode="reduced")
    diag = np.abs(np.diag(R))
    keep = diag > diag.max() * max(Xv.shape) * np.finfo(float).eps
    if (n_drop := int((~keep).sum())):
        print(f"    rank guard: dropped {n_drop} dependent column(s) "
              f"({X.shape[1]} → {int(keep.sum())})")
    return X.loc[:, keep]


def stabilize_design(X: pd.DataFrame, y: pd.Series, clusters: pd.Series):
    row_ok = (np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
              & np.isfinite(y.to_numpy(dtype=float))
              & clusters.notna().to_numpy())
    X, y, clusters = X.loc[row_ok], y.loc[row_ok], clusters.loc[row_ok]
    X = X.loc[:, X.std(ddof=0) > 0]
    X = X.loc[:, ~X.T.duplicated()]
    X = X.loc[:, X.sum(axis=0) != 1]
    return drop_dependent_columns(X).astype(float), y, clusters


def fit_ols_clustered(y: pd.Series, X: pd.DataFrame, clusters: pd.Series):
    return sm.OLS(y, X, hasconst=False).fit(
        cov_type="cluster", cov_kwds={"groups": clusters, "use_correction": True})


def centered_r2(res, n_params: int):
    n  = int(res.nobs)
    r2 = 1.0 - res.ssr / res.centered_tss
    return r2, (1.0 - (1.0 - r2) * (n - 1) / (n - n_params) if n > n_params else float("nan"))


def winsorize_cols(frame: pd.DataFrame, cols: list, mask: pd.Series, p: float = 0.01):
    out = frame.copy()
    for c in cols:
        if c in out.columns:
            # Cast to float first: some level vars are nullable Int64 and clipping them to a
            # fractional quantile bound raises. Estimate-preserving (cast to float downstream).
            out[c] = out[c].astype(float)
            lo, hi = out.loc[mask, c].quantile(p), out.loc[mask, c].quantile(1 - p)
            out[c] = out[c].clip(lower=lo, upper=hi)
    return out


def _stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


# ── Windowed sample ──────────────────────────────────────────────────────────────

def prepare_sample(lender_lists: pd.Series, window_years: int):
    df = pd.read_parquet(CONTRACTS)
    df = df[df["claude_is_debt_contract"] == "Y"]
    df = df[df[DV_SCORES].notna().all(axis=1)].copy()

    for sheet, cols in DV_SPECS:
        df[f"{sheet}_score_dummy"] = (df[cols] > 0).any(axis=1).astype(float)
    df["contract_year"] = df["tranche_active_date"].dt.year

    gaap, freeze = df["claude_GAAP_OVERRIDE_SCORE"], df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan
    # The two components — regressors in Panels C/D, unused in A/B.
    df["gaap_override"] = (gaap > 0).astype(float)
    df.loc[gaap.isna(), "gaap_override"] = np.nan
    df["freeze"] = (freeze > 0).astype(float)
    df.loc[freeze.isna(), "freeze"] = np.nan

    adopt = pd.to_datetime(df["adoption_date"], errors="coerce")
    df[POST] = np.where(adopt.isna(), np.nan,
                        (df["tranche_active_date"] >= adopt).astype(float))
    df = df[df[POST].notna()].copy()          # adopting firms only
    n_adopt = len(df)

    adopt = pd.to_datetime(df["adoption_date"], errors="coerce")
    in_window = ((df["tranche_active_date"] >= adopt - pd.DateOffset(years=window_years))
                 & (df["tranche_active_date"] <= adopt + pd.DateOffset(years=window_years)))
    df = df[in_window].copy()
    print(f"  ±{window_years}y window: {len(df):,} of {n_adopt:,} adopting-firm rows "
          f"| post=1: {int((df[POST] == 1).sum()):,}  post=0: {int((df[POST] == 0).sum()):,}")

    N, sic_digits = len(df), 2
    fe_iy, n_cells = make_industry_year_fe(df, sic_digits)
    while n_cells >= N - 5 and sic_digits > 1:
        sic_digits -= 1
        fe_iy, n_cells = make_industry_year_fe(df, sic_digits)
    X_fe   = fe_iy.loc[:, (fe_iy != 0).any(axis=0)]
    fe_bor = make_borrower_fe(df)

    df = df.join(lender_lists, on=MERGE_KEYS)
    df["lender_ids"] = df["lender_ids"].apply(lambda x: x if isinstance(x, list) else [])
    fe_lender = make_lender_multi_hot(df, df["lender_ids"])

    fe_labels = [f"Industry×Year FEs (SIC {sic_digits}-digit)", "Borrower FEs", "Lender FEs"]
    print(f"  FEs — IY={X_fe.shape[1]} Bor={fe_bor.shape[1]} Len={fe_lender.shape[1]}")
    return df, X_fe, fe_bor, fe_lender, fe_labels


# ── One DV column ────────────────────────────────────────────────────────────────

def run_column(df, X_fe, fe_bor, fe_lender, fe_labels, sheet, spec) -> list:
    dv = f"{sheet}_score_dummy"

    parents = [POST] + spec["determinants"] + CONTROLS
    finite  = np.isfinite(df[parents].to_numpy(dtype=float)).all(axis=1)
    mask    = pd.Series(finite, index=df.index) & df[dv].notna() & df["gvkey"].notna()

    # Winsorize BEFORE forming interactions so post × x inherits the winsorized parent.
    dfw = winsorize_cols(df, WINSOR_LEVEL_VARS, mask)
    for v in spec["interact_vars"]:
        dfw[interaction_name(v)] = dfw[POST] * dfw[v]

    X_full = pd.concat([dfw[spec["regressors"]].astype(float), X_fe, fe_bor, fe_lender], axis=1)
    X, y, cl = stabilize_design(X_full, dfw[dv], dfw["gvkey"])
    res = fit_ols_clustered(y, X, cl)
    r2c, adjc = centered_r2(res, X.shape[1])
    fe_counts = [sum(c.startswith(p) for c in X.columns) for p in ("iy_", "b_", "l_")]

    if (dropped := [v for v in spec["test_vars"] if v not in X.columns]):
        print(f"    ⚠ test variable(s) dropped by the rank guard: {', '.join(dropped)}")
    key = [interaction_name(v) for v in spec["acct"]]
    print(f"    {sheet:<8} N={len(y):,}  R²={r2c:.4f}  clusters={cl.nunique():,}  | " + "  ".join(
        f"{k}={res.params[k]:.3f}{_stars(res.pvalues[k])}" for k in key if k in res.params.index))

    return ([dv]
            + [f"{res.params[v]:.3f}{_stars(res.pvalues[v])}" if v in res.params.index else ""
               for v in spec["row_order"]]
            + ["", f"{len(y):,}", f"{r2c:.4f}", f"{adjc:.4f}"] + [str(c) for c in fe_counts])


def build_panel(df, X_fe, fe_bor, fe_lender, fe_labels, spec) -> pd.DataFrame:
    col_data = {f"({i})": run_column(df, X_fe, fe_bor, fe_lender, fe_labels, sheet, spec)
                for i, (sheet, _) in enumerate(DV_SPECS, start=1)}
    index = (["Dependent variable"]
             + [v.replace(f"{POST}_x_", f"{POST} × ") for v in spec["row_order"]]
             + ["", "N", "R²", "Adj. R²"] + fe_labels)
    return pd.DataFrame(col_data, index=index)


def run() -> None:
    lender_lists = load_lender_lists()
    cache, sheets = {}, {}
    for name, yrs, acct in PANELS:
        kind = "composite" if acct == COMPOSITE else "decomposed: " + " + ".join(acct)
        print(f"\n{name}  (±{yrs}y, {kind})")
        if yrs not in cache:                      # windowed sample + FEs reused across panels
            cache[yrs] = prepare_sample(lender_lists, yrs)
        df, X_fe, fe_bor, fe_lender, fe_labels = cache[yrs]
        sheets[name] = build_panel(df, X_fe, fe_bor, fe_lender, fe_labels, make_spec(acct))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        for name, tab in sheets.items():
            tab.to_excel(xw, sheet_name=name)
            ws = xw.sheets[name]
            ws.set_column(0, 0, 32)
            ws.set_column(1, tab.shape[1], 16)
    print(f"\nSaved → {OUT_FILE}  (sheets: {', '.join(sheets)})")


if __name__ == "__main__":
    run()
