"""
07_rq2_experience.py
RQ2: How does lender accounting and non-accounting experience shape covenant design?

This script emulates the early "full_regression_v1/v2" analyses under the current
(01–06) pipeline and its new variable names. The estimating equation is the RQ1
**Model 7** specification:

  DV  = ALL_score_dummy  (= 1 if ANY of the five claude_*_SCOREs > 0, else 0;
        identical to the RQ1 "ALL" dependent variable)
  FE  = Industry×Year (2-digit SIC × year) + Borrower + Lender (multi-hot)   [dense]
  X   = the 5 RQ1 determinants + 19 deal/firm controls, winsorized (1% both tails)
        on the estimation sample; SEs clustered by gvkey.

RQ2 layers lender *experience* regressors on top of this base. The test variables measure
the syndicate's exposure to irregularity events in its lenders' other borrower portfolios,
over a lookback window. The full 16-column table is produced at each of three lookback
windows — 36, 24 and 12 months — one worksheet each, named for the window. The window
selects the source columns' _12/_24/_36 suffix (from full_parent_event_selected_{w}.csv,
already merged into contracts.parquet by 03_contracts.py).

Seven event families are reported (see EVENTS):
  AEC  — accounting estimate changes            (count col est_nc_sum_max_aec_*)
  FR   — financial restatements                 (count col res_sum_adv1_max_fr_*)
  IC   — internal control weakness              (element-wise max of the auditor and
                                                 manager count cols ic_sum_a_ineff_max_ic_*
                                                 and ic_sum_m_ineff_max_ic_*; the two are
                                                 near-collinear (r ≈ 0.97–1.00) and share
                                                 one grouping column, so max avoids
                                                 double-counting overlapping weaknesses)
  GC   — going concern                          (count col aqrm_gc_sum_max_aqrm_*)
  LF   — late filing                            (count col aqrm_lf_sum_max_aqrm_*)
  MI   — material impairment                    (count col aqrm_mi_sum_max_aqrm_*)
  SP   — S&P default                            (count col sp_default_sum_max_sp_*)

Test-variable construction (see build_experience):
  Source columns, per relatedness r ∈ {u, r} and lender sample s ∈ {a, l}:
    {stem}_{r}{s}_{w}                     — count of events attributable to the syndicate
      For IC the stem field is a LIST of two count-column stems; build_experience takes
      their element-wise max (row-wise) to form a single IC count before bucketing.
    selected_grouping_{grp_root}_{r}{s}_{w} — True if the lender that experienced the
                                      events is a top-5 lender, False if not, None if none
  where  u = unrelated borrowers (different industry/geography), r = related borrowers,
         a = all syndicate lenders,       l = lead arrangers only.

  The `_max_` construction identifies a single "winner" lender per observation, so its
  event count lands entirely in EITHER the Top5 bucket OR the Non-Top5 bucket (the other
  is 0) — the two are never simultaneously positive. Grouping = None always coincides with
  a zero count (verified in all windows/suffixes), so those rows contribute 0 to both.
  Each bucket is then transformed as log(1 + x), matching the May full_regression_v2 runs.

  Yielding four test variables per event family, internally prefixed by that family
  (e.g. aec_NonTop5_Unrelated, fr_Top5_Related). In the OUTPUT TABLE the prefix is
  stripped, so all families share the same four rows (NonTop5_Unrelated, Top5_Unrelated,
  NonTop5_Related, Top5_Related); each column's family is named in the header row
  "Lender event from 36 months".

The table reports EVENTS × LENDER_SAMPLES columns. Within an event family the two columns
differ ONLY in which lenders the test variables are built from — everything else (DV, FEs,
determinants, controls) is held fixed:
  (1)  AEC, all lenders   — test vars from the _ua / _ra source columns
  (2)  AEC, lead lenders  — test vars from the _ul / _rl source columns
  (3)  FR,  all lenders         (9)  LF, all lenders
  (4)  FR,  lead lenders        (10) LF, lead lenders
  (5)  IC,  all lenders         (11) MI, all lenders
  (6)  IC,  lead lenders        (12) MI, lead lenders
  (7)  GC,  all lenders         (13) SP, all lenders
  (8)  GC,  lead lenders        (14) SP, lead lenders

Note the lead-lender source columns carry 35 genuinely missing counts (no lead arranger
identified). These are preserved as NaN — NOT coerced to 0 — so those rows drop from the
lead-lender columns, which therefore have a slightly smaller N.

The test variables are log-transformed and so are NOT winsorized (mirroring 05's
treatment of its logged controls).

Input:  data/contracts.parquet       (output of 03_contracts.py + 04_merge.py)
        data/dealscan_raw.parquet    (for lender_parent_id multi-hot FEs)
Output: output/tables/rq2_experience.xlsx  (sheets "36", "24", "12"; 16 columns each)

Runtime ≈ 25 min (42 regressions against ~4,000 FE columns). The DV and the FE matrices
are window-independent and are therefore built once and reused across all three sheets.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR.parent / "data"
OUT_DIR  = REPO_DIR.parent / "output" / "tables"
OUT_FILE = OUT_DIR / "rq2_experience.xlsx"

MERGE_KEYS = ["borrower_id", "tranche_active_date"]

# Dependent variable: reuse the RQ1 "ALL" dummy (1 if any of the five scores > 0).
DV_SHEET  = "ALL"
DV_SCORES = ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
             "claude_VAR_SCORE", "claude_RES_SCORE"]

# ── RQ2 lender-experience test variables ──────────────────────────────────────────
# Lookback windows for the event counts, sourced from full_parent_event_selected_{w}.csv
# (already merged into contracts.parquet by 03_contracts.py, suffixed _12/_24/_36).
# One worksheet per window, named for it; the full 16-column table is written to each.
WINDOWS = ["36", "24", "12"]

# Event families. Each entry is (prefix, count-column stem(s), grouping root, table label):
#   prefix   — internal variable-name prefix (keeps families from colliding)
#   stem     — root of the event-count column, i.e. {stem}_{r}{s}_{window}. May be a
#              LIST of stems for families that combine multiple count columns; in that
#              case build_experience takes the element-wise row-wise MAX of the columns
#              before bucketing.  Used for IC (auditor + manager).
#   grp_root — root of the selected_grouping_* column, i.e.
#              selected_grouping_{grp_root}_{r}{s}_{window}
#   label    — shown in the event header row of the output table
# For aec/fr the prefix and grouping root coincide. They do NOT for ic/aqrm: several
# count columns share one grouping column (ic_sum_a_ineff / ic_sum_m_ineff both use
# selected_grouping_ic_*; likewise aqrm_gc/lf/mi share selected_grouping_aqrm_*), which
# is why the two are kept as separate fields.
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

# The two lender samples. Value = the {r}{s} source suffix.
# Output columns are EVENTS × LENDER_SAMPLES, numbered (1)…(4) left to right.
LENDER_SAMPLES = [
    ("All lenders",  "a"),
    ("Lead lenders", "l"),
]

# The four test variables, in output row order. Constructed in build_experience();
# log(1+x), hence excluded from winsorization. Internally each carries its event prefix
# (aec_, fr_, …) so that event families cannot collide.
EXPERIENCE_BASE = [
    "NonTop5_Unrelated",
    "Top5_Unrelated",
    "NonTop5_Related",
    "Top5_Related",
]

def exp_names(prefix: str) -> list:
    """Internal (prefixed) names of the four test variables for one event family."""
    return [f"{prefix}_{v}" for v in EXPERIENCE_BASE]

# In the output table the prefix is stripped, so every event family reuses these same
# four rows — the column's event is identified by the "Lender event from N months"
# header row instead.
EXPERIENCE_DISPLAY = EXPERIENCE_BASE

# The RQ1 determinants (same construction as 06_rq1_determinants.py) — kept in lock-step
# with 06: credit quality enters as the four mutually-exclusive bucket dummies built on the
# FISD-supplemented, unrestricted "_all" rating (03 §2c), with ig_grade (investment grade,
# BBB- or above) the OMITTED reference category.
DETERMINANTS = ["accounting_policy", "offbslease", "BB_grade", "B_grade", "CCC_below", "non_rated_suppl_all", "relationship_freq"]

# Deal + borrower-level controls (identical to Model 7 in 06), incl. the two FISD bond controls.
CONTROLS = [
    "maturity", "log_lender_count", "log_interest", "log_deal_amount", "perf_pricing",
    "fin_covenant_count", "gen_covenant_count", "secured",
    "size", "profitability", "bsfixed", "liabilities", "logage", "btm", "capex",
    "loss", "rand", "divyield",
    "log_bond_count", "bond_proceeds_scaled",
]
# Non-logged, non-dummy level variables winsorized at 1% both tails on the estimation
# sample (mirrors 06's WINSOR_LEVEL_VARS).
WINSOR_LEVEL_VARS = [
    "offbslease",
    "fin_covenant_count", "gen_covenant_count",
    "profitability", "bsfixed", "liabilities", "btm", "capex", "rand", "divyield",
    "bond_proceeds_scaled",
]


# ── Fixed-effect builders (dense; copied from 05) ─────────────────────────────────

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


# ── Design-matrix cleaner (copied from 05) ────────────────────────────────────────

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
    borrower dummies, hence sum(iy) − sum(borrower) = 0.  A rank-deficient matrix this size
    (~10,000 × ~4,000) is handed by statsmodels' default `pinv` path to LAPACK's `gesdd` SVD,
    which INTERMITTENTLY FAILS TO CONVERGE — `numpy.linalg.LinAlgError: SVD did not converge`.
    It is a numerical coin-flip, not a property of any particular column: it surfaced in the
    no-controls variant (07b) on the 12-month ic_m model after 38 clean fits.  This guard makes
    the solve well-posed so a re-run cannot die at random.

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


def fit_ols_clustered(y: pd.Series, X: pd.DataFrame, clusters: pd.Series):
    model = sm.OLS(y, X, hasconst=False)
    return model.fit(cov_type="cluster",
                     cov_kwds={"groups": clusters, "use_correction": True})


# ── Output formatting (copied from 05) ────────────────────────────────────────────

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
    bounds computed on the rows selected by `mask` (the regression sample)."""
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
                 master_regressors: list, dv_name: str, sample_label: str,
                 event_label: str) -> list:
    """Return values list: DV name, event family, lender sample, then coef/t-stat rows
    aligned to master_regressors, then footer rows (N, R², Adj. R², one row per FE type).

    `master_regressors` are the INTERNAL (prefixed) names used to look up coefficients;
    the row labels shown in the table are set separately by the caller."""
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


# ── Shared inputs (copied from 05) ────────────────────────────────────────────────

def build_experience(df: pd.DataFrame, prefix: str, stem, grp_root: str,
                     sample_suf: str, window: str) -> pd.DataFrame:
    """Build the four log(1+x) test variables for one event family and lender sample.

    prefix/stem/grp_root: the event family (see EVENTS).  `stem` may be a single string
      or a LIST of stems; for a list, the event count is the element-wise row-wise MAX
      of the listed columns (used for IC = max of auditor + manager weakness counts).
    sample_suf:           "a" (all lenders) or "l" (lead arrangers only).
    window:               "12", "24" or "36" — the event-count lookback window.

    For each relatedness bucket, the observation's event count is assigned to the Top5
    column when the winner lender is a top-5 lender and to the Non-Top5 column when it is
    not; the other column gets 0. A None grouping means no event occurred (count is always
    0 there — verified for every event/window/suffix), so both columns get 0. Genuinely
    missing counts (NaN — no lead arranger identified) are preserved as NaN in BOTH
    columns so the row drops from the regression rather than being read as a true zero.
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


# ── Base Model 7 estimation ───────────────────────────────────────────────────────

def prepare_sample(df_full: pd.DataFrame, lender_lists: pd.Series):
    """Apply the sample filters, build the DV and the (sample-invariant) FE matrices.
    These are identical across the two lender-sample columns, so they are built once."""
    dv = f"{DV_SHEET}_score_dummy"
    print(f"\n{'#' * 60}\n#  RQ2 — DV = {dv}  |  windows: {', '.join(WINDOWS)}\n{'#' * 60}")

    # Sample filters (identical to 05)
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
    """Estimate the Model 7 spec with one event family's test variables built from one
    lender sample and lookback window, and return the formatted output column."""
    prefix, stem, grp_root, event_label = event
    experience = exp_names(prefix)
    regressors = experience + DETERMINANTS + CONTROLS
    IY_LABEL, BOR_LABEL, LEN_LABEL = fe_labels

    print(f"\n{'=' * 60}\n  [{window}mo] Model {col_name}: {prefix} × {sample_label} "
          f"(source suffix _{sample_suf})\n{'=' * 60}")

    # Build the four test variables for this event × lender sample and attach them
    print(f"  building {prefix} test variables:")
    df = df.copy()
    df[experience] = build_experience(df, prefix, stem, grp_root, sample_suf, window)

    def _sample_mask(frame, cols):
        finite = np.isfinite(frame[cols].to_numpy(dtype=float)).all(axis=1)
        return (pd.Series(finite, index=frame.index)
                & frame[dv].notna() & frame["gvkey"].notna())

    m7_mask = _sample_mask(df, regressors)
    print(f"\n  Estimation sample (all regressors non-missing): {int(m7_mask.sum()):,}")
    # Test variables are log(1+x) → not winsorized, matching 05's treatment of logged vars.
    print("  winsorizing level variables at 1% on this sample:")
    dfw = winsorize_cols(df, WINSOR_LEVEL_VARS, m7_mask)

    X_full = pd.concat([dfw[regressors].astype(float), X_fe, fe_bor, fe_lender], axis=1)
    X, y_clean, cl_clean = stabilize_design(X_full, dfw[dv], dfw["gvkey"])
    res = fit_ols_clustered(y_clean, X, cl_clean)

    iy_cols = [c for c in X.columns if c.startswith("iy_")]
    b_cols  = [c for c in X.columns if c.startswith("b_")]
    l_cols  = [c for c in X.columns if c.startswith("l_")]
    fe_counts = {IY_LABEL: len(iy_cols), BOR_LABEL: len(b_cols), LEN_LABEL: len(l_cols)}

    print(f"  N = {len(y_clean):,}  |  R² = {res.rsquared:.4f}  |  Adj. R² = {res.rsquared_adj:.4f}")
    print(f"  IY FEs: {len(iy_cols)}  |  Borrower FEs: {len(b_cols)}  |  Lender FEs: {len(l_cols)}")
    print(f"  Unique clusters (gvkey): {cl_clean.nunique():,}")
    for v in experience:
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
    print("⚠  REMINDER: R²/Adj. R² reported below are UNCENTERED (hasconst=False). 06/06b/06c "
          "switched to CENTERED Adj. R² on 2026-08-11 — update this script to match before using its R².")
    print("Loading contracts …")
    df_full = pd.read_parquet(DATA_DIR / "contracts.parquet")
    print(f"  {len(df_full):,} rows × {df_full.shape[1]} cols")

    lender_lists = load_lender_lists()
    # The DV and the FE matrices do not depend on the lookback window, so they are built
    # once and reused across every window's table.
    df, dv, X_fe, fe_bor, fe_lender, fe_labels = prepare_sample(df_full, lender_lists)

    # Row labels use the DISPLAY names (prefix stripped) so that every event family
    # reuses the same four test-variable rows; the event is identified per-column by
    # the "Lender event from N months" header row.
    display_regressors = EXPERIENCE_DISPLAY + DETERMINANTS + CONTROLS
    footer_labels      = ["", "N", "R²", "Adj. R²"] + fe_labels

    tables = {}
    for window in WINDOWS:
        print(f"\n{'#' * 60}\n#  LOOKBACK WINDOW: {window} months  "
              f"(worksheet '{window}')\n{'#' * 60}")

        # Columns are EVENTS × LENDER_SAMPLES, numbered (1)…(16) left to right:
        # (1) AEC/all, (2) AEC/lead, (3) FR/all, (4) FR/lead, …
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
