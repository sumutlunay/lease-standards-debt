"""
07c_rq2_table5.py  →  output/tables/table5.xlsx   (Table 5)

RQ2 lender-experience results, WITH-CONTROLS specification (the full 07 Model-7 spec), in the same
two-panel layout as Table 4. This is the controlled counterpart to 07_rq2_table4.py: the
regression ADDS the five RQ1 determinants + the 20 deal/firm controls back to the right-hand side.

  • 12-MONTH window only.
  • RHS = the four experience test variables (NonTop5_Unrelated, Top5_Unrelated, NonTop5_Related,
    Top5_Related) + the 5 determinants + the 20 controls + the Model-7 fixed effects — the FULL
    07 spec. Non-logged, non-dummy level variables are winsorized 1% both tails on the estimation
    sample, exactly as in 07 (see WINSOR_LEVEL_VARS).
  • Only the four TEST-VARIABLE rows are tabulated; the determinant/control coefficients are
    estimated but NOT shown (nuisance controls here).
  • Sample = 07's Model-7 sample (all regressors + controls non-missing) → IDENTICAL rows to
    table4, so Table 4 vs Table 5 differ ONLY in whether the controls are in the RHS.
  • Coefficients + significance stars only — NO t-statistics.
  • R² / Adj. R² reported CENTERED (about the DV mean; see centered_r2), matching table2/table3.
  • Two stacked panels in ONE sheet:
        Panel A — all syndicate lenders' experience   (source suffix _{u,r}a)
        Panel B — lead arrangers' experience          (source suffix _{u,r}l)
    Each panel: 7 event-family columns (AEC, FR, IC, GC, LF, MI, SP) × the 4 test-var rows.

DV, FE structure, event families and the log(1+x) test-variable construction are identical to
07 (see 07_rq2_experience.py for the full documentation of EVENTS and build_experience). SEs are
clustered by gvkey (needed for the stars, even though the t-stats are not printed).

Input:  data/contracts.parquet       (output of 03_contracts.py + 04_merge.py)
        data/dealscan_raw.parquet    (for lender_parent_id multi-hot FEs)
Output: output/tables/table5.xlsx    (single sheet 'Table 5', two panels)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_DIR = Path(__file__).resolve().parent.parent  # code/ — this script lives in code/exploratory/
DATA_DIR = REPO_DIR.parent / "data"
OUT_DIR  = REPO_DIR.parent / "output" / "tables"
OUT_FILE = OUT_DIR / "table5.xlsx"
SHEET    = "Table 5"

MERGE_KEYS = ["borrower_id", "tranche_active_date"]

DV_SHEET  = "ALL"
DV_SCORES = ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
             "claude_VAR_SCORE", "claude_RES_SCORE"]

WINDOW = "12"   # 12-month lookback only

# Event families (identical to 07): (prefix, count-stem(s), grouping root, label).
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

# Two lender samples → two panels. Value = the {r}{s} source suffix's sample code.
LENDER_SAMPLES = [
    ("Panel A: All lenders' experience (12-month window)",  "a"),
    ("Panel B: Lead lenders' experience (12-month window)", "l"),
]

EXPERIENCE_BASE = [
    "NonTop5_Unrelated",
    "Top5_Unrelated",
    "NonTop5_Related",
    "Top5_Related",
]

def exp_names(prefix: str) -> list:
    return [f"{prefix}_{v}" for v in EXPERIENCE_BASE]

# The full 07 Model-7 controls — here these ARE regressors (estimated but NOT tabulated).
DETERMINANTS = ["accounting_policy", "offbslease", "num_rating_suppl_all", "non_rated_suppl_all", "relationship_freq"]
CONTROLS = [
    "maturity", "log_lender_count", "log_interest", "log_deal_amount", "perf_pricing",
    "fin_covenant_count", "gen_covenant_count", "secured",
    "size", "profitability", "bsfixed", "liabilities", "logage", "btm", "capex",
    "loss", "rand", "divyield",
    "log_bond_count", "bond_proceeds_scaled",
]
# Non-logged, non-dummy level variables winsorized 1% both tails on the estimation sample (as in 07).
WINSOR_LEVEL_VARS = [
    "offbslease",
    "fin_covenant_count", "gen_covenant_count",
    "profitability", "bsfixed", "liabilities", "btm", "capex", "rand", "divyield",
    "bond_proceeds_scaled",
]


def event_code(event: tuple) -> str:
    """Short column code from the event label, e.g. 'AEC: …' → 'AEC'."""
    return event[3].split(":")[0].strip()


# ── Fixed-effect builders (identical to 07) ───────────────────────────────────────

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


# ── Design cleaner + estimation (identical to 07) ─────────────────────────────────

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
    Adj = 1 − (1−R²_c)(n−1)/(n−k). See 06's centered_r2."""
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
    """Return a copy of `frame` with each column in `cols` winsorized at [p, 1-p], bounds
    computed on the rows selected by `mask` (the estimation sample). Mirrors 07."""
    out = frame.copy()
    for c in cols:
        if c not in out.columns:
            continue
        lo = out.loc[mask, c].quantile(p)
        hi = out.loc[mask, c].quantile(1 - p)
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out


# ── Test-variable construction (identical to 07) ──────────────────────────────────

def build_experience(df: pd.DataFrame, prefix: str, stem, grp_root: str,
                     sample_suf: str, window: str) -> pd.DataFrame:
    out = {}
    for rel_code, rel_name in [("u", "Unrelated"), ("r", "Related")]:
        suf = f"{rel_code}{sample_suf}"
        if isinstance(stem, (list, tuple)):
            src_cols = [f"{s}_{suf}_{window}" for s in stem]
            val = df[src_cols].max(axis=1, skipna=False).astype(float)
        else:
            val = df[f"{stem}_{suf}_{window}"].astype(float)
        grp = df[f"selected_grouping_{grp_root}_{suf}_{window}"]

        top5    = np.where(grp.eq(True),  val, 0.0)
        nontop5 = np.where(grp.eq(False), val, 0.0)
        missing = val.isna().to_numpy()
        top5[missing]    = np.nan
        nontop5[missing] = np.nan

        out[f"{prefix}_NonTop5_{rel_name}"] = np.log1p(nontop5)
        out[f"{prefix}_Top5_{rel_name}"]    = np.log1p(top5)
    return pd.DataFrame(out, index=df.index)[exp_names(prefix)]


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


# ── Base sample + FE matrices (identical to 07's prepare_sample) ──────────────────

def prepare_sample(df_full: pd.DataFrame, lender_lists: pd.Series):
    dv = f"{DV_SHEET}_score_dummy"
    print(f"\n{'#' * 60}\n#  RQ2 Table 4 — DV = {dv}  |  window: {WINDOW}mo  |  no controls\n{'#' * 60}")

    df = df_full[df_full["claude_is_debt_contract"] == "Y"].copy()
    print(f"  {len(df):,} rows after keeping claude_is_debt_contract == Y")
    df = df[df[DV_SCORES].notna().all(axis=1)].copy()
    print(f"  {len(df):,} rows after dropping rows missing any claude score")

    df[dv]              = (df[DV_SCORES] > 0).any(axis=1).astype(float)
    df["contract_year"] = df["tranche_active_date"].dt.year
    gaap   = df["claude_GAAP_OVERRIDE_SCORE"]
    freeze = df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan

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
    return df, dv, X_fe, fe_bor, fe_lender, fe_labels


# ── One no-controls regression (one event family × one lender sample) ─────────────

def run_experience_model(df, dv, X_fe, fe_bor, fe_lender, fe_labels,
                         event: tuple, sample_suf: str) -> list:
    """Return the output column (event label + 4 coefs/stars + footer) for one event family
    and lender sample. RHS = 4 test variables + FE only; sample pinned to 07's Model-7 sample."""
    prefix, stem, grp_root, event_label = event
    experience = exp_names(prefix)
    regressors = experience + DETERMINANTS + CONTROLS   # FULL 07 Model-7 spec
    IY_LABEL, BOR_LABEL, LEN_LABEL = fe_labels
    print(f"\n{'=' * 60}\n  [{WINDOW}mo] {event_code(event)} × suffix _{sample_suf}\n{'=' * 60}")

    df = df.copy()
    df[experience] = build_experience(df, prefix, stem, grp_root, sample_suf, WINDOW)

    # 07 Model-7 estimation sample: experience + determinants + controls all non-missing.
    finite    = np.isfinite(df[regressors].to_numpy(dtype=float)).all(axis=1)
    m7_mask   = pd.Series(finite, index=df.index) & df[dv].notna() & df["gvkey"].notna()
    sub       = df.index[m7_mask]
    print(f"  sample (07 Model-7 rows, all regressors+controls non-missing): {len(sub):,}")

    # Model = experience + determinants + controls + FE (the full 07 spec). Winsorize the level
    # variables at 1% on the estimation sample (test vars are log(1+x) → not winsorized).
    dfw = winsorize_cols(df, WINSOR_LEVEL_VARS, m7_mask)
    X_full = pd.concat([dfw.loc[sub, regressors].astype(float),
                        X_fe.loc[sub], fe_bor.loc[sub], fe_lender.loc[sub]], axis=1)
    X, y_clean, cl_clean = stabilize_design(X_full, dfw.loc[sub, dv], dfw.loc[sub, "gvkey"])
    res = fit_ols_clustered(y_clean, X, cl_clean)
    r2c, adjc = centered_r2(res, X.shape[1])

    iy = sum(c.startswith("iy_") for c in X.columns)
    b  = sum(c.startswith("b_")  for c in X.columns)
    l  = sum(c.startswith("l_")  for c in X.columns)

    print(f"  N = {len(y_clean):,}  |  R² = {r2c:.4f} (centered)  |  clusters = {cl_clean.nunique():,}")
    for v in experience:
        if v in res.params.index:
            print(f"    {v:<24} {res.params[v]:>8.4f}{_stars(res.pvalues[v]):<3}")

    # Column values: event label, 4 coef/stars (no t-stat), blank, N, R², Adj.R², FE counts.
    col = [event_label]
    for v in experience:
        col.append(f"{res.params[v]:.3f}{_stars(res.pvalues[v])}" if v in res.params.index else "")
    col += ["", f"{len(y_clean):,}", f"{r2c:.4f}", f"{adjc:.4f}", str(iy), str(b), str(l)]
    return col


def build_panel(df, dv, X_fe, fe_bor, fe_lender, fe_labels, sample_suf: str) -> pd.DataFrame:
    """One panel: 7 event columns for a single lender sample."""
    cols = {event_code(ev): run_experience_model(
                df, dv, X_fe, fe_bor, fe_lender, fe_labels, ev, sample_suf)
            for ev in EVENTS}
    index = (["Lender event"] + EXPERIENCE_BASE
             + ["", "N", "R²", "Adj. R²"] + fe_labels)
    return pd.DataFrame(cols, index=index)[[event_code(ev) for ev in EVENTS]]


# ── Main ──────────────────────────────────────────────────────────────────────────

def run():
    print("Loading contracts …")
    df_full = pd.read_parquet(DATA_DIR / "contracts.parquet")
    print(f"  {len(df_full):,} rows × {df_full.shape[1]} cols")

    lender_lists = load_lender_lists()
    df, dv, X_fe, fe_bor, fe_lender, fe_labels = prepare_sample(df_full, lender_lists)

    panels = []
    for title, sample_suf in LENDER_SAMPLES:
        print(f"\n{'#' * 60}\n#  {title}\n{'#' * 60}")
        panel = build_panel(df, dv, X_fe, fe_bor, fe_lender, fe_labels, sample_suf)
        panels.append((title, panel))

    OUT_DIR.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        row = 0
        ws  = None
        for title, panel in panels:
            panel.to_excel(xw, sheet_name=SHEET, startrow=row + 1)
            ws = xw.sheets[SHEET]
            ws.write(row, 0, title)                    # panel title above its table
            row += 1 + (len(panel.index) + 1) + 2      # title + (header + rows) + 2 blank
        ws.set_column(0, 0, 22)
        ws.set_column(1, len(EVENTS), 14)

    print(f"\nSaved → {OUT_FILE}  (sheet: {SHEET}; two panels × {len(EVENTS)} event columns)")


if __name__ == "__main__":
    run()
