"""
09_diagnostics.py
Pre-analysis diagnostics for the RQ2/RQ3 extensions.  Nothing is written to disk;
each diagnostic prints to the console so we can eyeball before deciding on a
persistent output.  Add more diagnostics as separate functions and dispatch from
__main__.

────────────────────────────────────────────────────────────────────────────────
DIAGNOSTIC 1 — rq2_distributions()
────────────────────────────────────────────────────────────────────────────────
Distribution of the 84 RQ2 lender-experience test variables:

  4 dimensions (NonTop5_Unrelated, Top5_Unrelated, NonTop5_Related, Top5_Related)
  × 7 event families (AEC, FR, IC, GC, LF, MI, SP — IC = max of auditor + manager)
  × 3 lookback windows (36, 24, 12 months)
  = 84 variables, all in the log(1+x) regression form used in 07_rq2_experience.py.

The lender-sample axis is COLLAPSED to the primary (all-lender, "_a" suffix). Lead-
lender variants ("_l") are easy to switch — see LENDER_SUFFIX below.

Sample: the RQ2 Model 7 estimation sample, reproduced here exactly as in 07:
claude_is_debt_contract == "Y", every claude_*_SCORE non-missing, all 5 determinants
and 21 controls non-missing (incl. the two FISD bond controls), gvkey non-missing.
Credit quality enters as the four mutually-exclusive bucket dummies (BB_grade, B_grade,
CCC_below, non_rated_suppl_all) built on the FISD-supplemented num_rating_suppl_all, with
ig_grade (investment grade) the omitted reference — matching 06/07.

Stats reported (per variable, matching 05_descriptives.py):
    N, min, p10, p25, p50, p75, p90, max, mean, std.

Note: because the underlying event counts are 0 for most observations, the log(1+x)
transform leaves the median (and often the 75th percentile) at exactly 0.  This is
the diagnostic point — mean/std alone would be misleading.

Input:  data/contracts.parquet
Output: output/tables/diagnostic1_rq2_distributions.xlsx  (sheets "36", "24", "12";
        one row per event × dim; 28 rows per sheet.  Also printed to console.)

────────────────────────────────────────────────────────────────────────────────
DIAGNOSTIC 2 — rq2_rq1_correlations()
────────────────────────────────────────────────────────────────────────────────
Correlation matrix between the 84 RQ2 event variables (as above) and the 31 RQ1
variables (5 DVs + 5 determinants + 21 controls).  Rectangular 84 × 31.

Both sides in REGRESSION FORM:
  • Event vars: log(1+x)  (as in 07)
  • DVs and dummies: as-is
  • Continuous RQ1 vars: winsorized 1%/99% on the same sample (matches 06's
    correlations sheet); logged vars kept logged.

Two sheets, both 84 × 31, matching 06's coefficient-plus-stars format:
    pearson   — Pearson correlations (what a linear regression absorbs)
    spearman  — Spearman rank correlations (robust to zero-inflation)
Stars: *** p<0.01, ** p<0.05, * p<0.10.

Also prints the top-10 |correlations| for the three robust 12mo Top5_Related cells
(IC / LF / MI) so the "which control absorbs the effect?" question is answerable
straight from the console.

Input:  data/contracts.parquet
Output: output/tables/diagnostic2_rq2_rq1_correlations.xlsx  (sheets "pearson",
        "spearman"; 84 rows × 31 cols each).
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR.parent / "data"
OUT_DIR  = REPO_DIR.parent / "output" / "tables"
OUT_FILE  = OUT_DIR / "diagnostic1_rq2_distributions.xlsx"
OUT_FILE2 = OUT_DIR / "diagnostic2_rq2_rq1_correlations.xlsx"

# ── Constants mirroring 07_rq2_experience.py ─────────────────────────────────────
WINDOWS = ["36", "24", "12"]

# (prefix, count-column stem(s), grouping-column root, table label) — same as 07.
# `stem` may be a LIST; build_experience_one then takes the element-wise row-wise MAX.
# IC = max(auditor, manager) — consolidated to mirror 07 (7 families, not the old 8).
EVENTS = [
    ("aec",     "est_nc_sum_max_aec",    "aec",  "AEC"),
    ("fr",      "res_sum_adv1_max_fr",   "fr",   "FR"),
    ("ic",      ["ic_sum_a_ineff_max_ic",
                 "ic_sum_m_ineff_max_ic"], "ic", "IC"),
    ("aqrm_gc", "aqrm_gc_sum_max_aqrm",  "aqrm", "GC"),
    ("aqrm_lf", "aqrm_lf_sum_max_aqrm",  "aqrm", "LF"),
    ("aqrm_mi", "aqrm_mi_sum_max_aqrm",  "aqrm", "MI"),
    ("sp",      "sp_default_sum_max_sp", "sp",   "SP"),
]

# Which lender sample to describe.  "a" = all lenders (primary); "l" = lead arrangers
# only (a few rows drop for missing lead-arranger counts). Sample size is derived, not
# hardcoded — it tracks 07's Model 7 base for the chosen suffix.
LENDER_SUFFIX = "a"

DIMS = ["NonTop5_Unrelated", "Top5_Unrelated", "NonTop5_Related", "Top5_Related"]

# Sample filters (identical to 07's Model 7 base).
DV_SCORES = ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
             "claude_VAR_SCORE", "claude_RES_SCORE"]
# Kept in lock-step with 06/07: _suppl_all rating BUCKETS (ig_grade = omitted reference)
# + the two FISD bond controls.
RATING_BUCKETS = ["BB_grade", "B_grade", "CCC_below", "non_rated_suppl_all"]
DETERMINANTS = ["accounting_policy", "offbslease"] + RATING_BUCKETS + ["relationship_freq"]
CONTROLS = [
    "maturity", "log_lender_count", "log_interest", "log_deal_amount", "perf_pricing",
    "fin_covenant_count", "gen_covenant_count", "is_covenant_ratio", "secured",
    "size", "profitability", "bsfixed", "liabilities", "logage", "btm", "capex",
    "loss", "rand", "divyield",
    "log_bond_count", "bond_proceeds_scaled",
]
COVENANT_RATIO_FILL = "is_covenant_ratio"

STAT_COLS = ["min", "p10", "p25", "p50", "p75", "p90", "max", "mean", "std"]


# ── Helpers ──────────────────────────────────────────────────────────────────────

def build_experience_one(df: pd.DataFrame, stem, grp_root: str,
                         sample_suf: str, window: str) -> pd.DataFrame:
    """Build the four log(1+x) test variables for one event family × lender sample × window.

    Byte-identical to 07_rq2_experience.build_experience() minus the internal
    event-prefix (dropped here since 09 prints event as a separate column). `stem` may be a
    single string or a LIST of stems; for a list the count is the element-wise row-wise MAX
    of the listed columns (used for IC = max of auditor + manager weakness counts).
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

        # .eq(True) / .eq(False) on the object dtype gives False for None → 0 in both buckets.
        top5    = np.where(grp.eq(True),  val, 0.0)
        nontop5 = np.where(grp.eq(False), val, 0.0)

        # Preserve genuine NaN counts (np.where above would map them to 0).
        missing = val.isna().to_numpy()
        top5[missing]    = np.nan
        nontop5[missing] = np.nan

        out[f"NonTop5_{rel_name}"] = np.log1p(nontop5)
        out[f"Top5_{rel_name}"]    = np.log1p(top5)

    return pd.DataFrame(out, index=df.index)[DIMS]


def compute_stats(s: pd.Series) -> dict:
    s = s.dropna().astype(float)
    return {
        "N":    int(s.shape[0]),
        "min":  s.min(),
        "p10":  s.quantile(0.10),
        "p25":  s.quantile(0.25),
        "p50":  s.quantile(0.50),
        "p75":  s.quantile(0.75),
        "p90":  s.quantile(0.90),
        "max":  s.max(),
        "mean": s.mean(),
        "std":  s.std(ddof=1),
    }


def prepare_sample(df_full: pd.DataFrame) -> pd.DataFrame:
    """Reproduce 07's Model 7 estimation sample (matches 07's all-lender base)."""
    df = df_full[df_full["claude_is_debt_contract"] == "Y"].copy()
    df = df[df[DV_SCORES].notna().all(axis=1)].copy()

    dv = "ALL_score_dummy"
    df[dv] = (df[DV_SCORES] > 0).any(axis=1).astype(float)

    gaap   = df["claude_GAAP_OVERRIDE_SCORE"]
    freeze = df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan

    df[COVENANT_RATIO_FILL] = df[COVENANT_RATIO_FILL].fillna(0)

    need   = [dv] + DETERMINANTS + CONTROLS
    finite = np.isfinite(df[need].to_numpy(dtype=float)).all(axis=1)
    mask   = pd.Series(finite, index=df.index) & df["gvkey"].notna()
    return df[mask].copy()


# ── Diagnostic 1: RQ2 distributions ──────────────────────────────────────────────

def rq2_distributions():
    """Descriptive statistics for the 84 RQ2 lender-experience test variables."""
    print("Loading contracts …")
    df_full = pd.read_parquet(DATA_DIR / "contracts.parquet")
    print(f"  {len(df_full):,} rows × {df_full.shape[1]} cols")

    smp = prepare_sample(df_full)
    sample_label = "all lenders" if LENDER_SUFFIX == "a" else "lead arrangers"
    print(f"  {len(smp):,} rows in RQ2 Model 7 estimation sample "
          f"(matches 07's base; describing {sample_label}, suffix '_{LENDER_SUFFIX}').")

    # Build the event series and compute stats.
    rows = []
    for window in WINDOWS:
        for prefix, stem, grp_root, event_label in EVENTS:
            exp = build_experience_one(smp, stem, grp_root, LENDER_SUFFIX, window)
            for dim in DIMS:
                stats = compute_stats(exp[dim])
                rows.append({
                    "window": window,
                    "event":  event_label,
                    "dim":    dim,
                    **stats,
                })

    out = pd.DataFrame(rows)
    out = out[["window", "event", "dim", "N"] + STAT_COLS]
    out[STAT_COLS] = out[STAT_COLS].round(4)
    out["N"] = out["N"].astype(int)

    print(f"\n{len(out)} rows = {len(WINDOWS)} windows × {len(EVENTS)} events × {len(DIMS)} dims. "
          f"All variables in log(1+x) regression form (as in 07).")
    print("Stats mirror 05_descriptives.py.  N below is the count of non-missing "
          "observations for that variable in the 07 estimation sample.\n")

    with pd.option_context(
        "display.width", 220,
        "display.max_rows", 120,
        "display.max_columns", 20,
        "display.colheader_justify", "right",
    ):
        for window in WINDOWS:
            print(f"\n{'=' * 100}")
            print(f"  LOOKBACK WINDOW: {window} months")
            print(f"{'=' * 100}")
            sub = out[out["window"] == window].drop(columns=["window"])
            print(sub.to_string(index=False))

    # ── Write xlsx (undated; freeze with a _YYYY-MM-DD suffix when sharing) ───
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_desc = ("all lenders (_a)" if LENDER_SUFFIX == "a"
                   else "lead arrangers (_l)")
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        for window in WINDOWS:
            sub = out[out["window"] == window].drop(columns=["window"])
            sub.to_excel(xw, sheet_name=window, index=False, startrow=2)
            ws = xw.sheets[window]
            ws.write(0, 0,
                     f"RQ2 lender-experience test variables — {window}-month lookback "
                     f"| sample: {len(smp):,} obs (RQ2 Model 7 base) | lender sample: {sample_desc}")
            ws.write(1, 0,
                     "Values in log(1+x) regression form (as in 07). "
                     "Stats mirror 05_descriptives.py.")

    print(f"\nSaved → {OUT_FILE}  (sheets: {', '.join(WINDOWS)})")


# ── Diagnostic 2: correlations of RQ2 event vars with RQ1 variables ──────────────

# RQ1 dependent variables (five DV dummies, per 05/06).
DV_SPECS = [
    ("SLB_score_dummy",     ["claude_SLB_SCORE"]),
    ("SYN_score_dummy",     ["claude_SYN_SCORE"]),
    ("OPL_score_dummy",     ["claude_OPL_SCORE"]),
    ("VAR-RES_score_dummy", ["claude_VAR_SCORE", "claude_RES_SCORE"]),
    ("ALL_score_dummy",     DV_SCORES),
]

# Continuous variables winsorized 1%/99% in 06 (matches 06's correlations sheet).
WINSOR_VARS = [
    "offbslease",
    "fin_covenant_count", "gen_covenant_count", "is_covenant_ratio",
    "profitability", "bsfixed", "liabilities", "btm", "capex", "rand", "divyield",
    "bond_proceeds_scaled",
]


def _corr_star(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def _corr_p(func, x: np.ndarray, y: np.ndarray):
    """Return (coef, p-value); handles both new scipy result objects and old tuples,
    and returns (NaN, NaN) if either side has zero variance."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan
    xv, yv = x[ok], y[ok]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return np.nan, np.nan
    res = func(xv, yv)
    try:
        return float(res.statistic), float(res.pvalue)
    except AttributeError:
        return float(res[0]), float(res[1])


def _fmt_corr(r: float, p: float) -> str:
    if not np.isfinite(r):
        return ""
    s = f"{r:.2f}"
    if s == "-0.00":
        s = "0.00"
    return s + _corr_star(p)


def rq2_rq1_correlations():
    """Diagnostic 2: 84 × 31 correlation matrix (RQ2 events × RQ1 vars),
    Pearson and Spearman.  Prints top correlates for the three robust 12mo cells."""
    print("Loading contracts …")
    df_full = pd.read_parquet(DATA_DIR / "contracts.parquet")
    print(f"  {len(df_full):,} rows × {df_full.shape[1]} cols")

    smp = prepare_sample(df_full)
    print(f"  {len(smp):,} rows in the RQ2 Model 7 estimation sample.")

    # ── Build the 5 DV dummies ────────────────────────────────────────────────
    for dv, cols in DV_SPECS:
        smp[dv] = (smp[cols] > 0).any(axis=1).astype(float)

    # ── Winsorize continuous RQ1 vars 1%/99% on this sample (matches 06) ──────
    print(f"\nWinsorizing {len(WINSOR_VARS)} continuous RQ1 variables at 1%/99% on the estimation sample:")
    for c in WINSOR_VARS:
        lo, hi = smp[c].quantile(0.01), smp[c].quantile(0.99)
        n_clip = int(((smp[c] < lo) | (smp[c] > hi)).sum())
        smp[c] = smp[c].clip(lower=lo, upper=hi)
        print(f"    {c:<20} [{lo:.4f}, {hi:.4f}]  ({n_clip} obs clipped)")

    # ── Build the event variables in regression form ──────────────────────────
    n_event_vars = len(WINDOWS) * len(EVENTS) * len(DIMS)
    print(f"\nBuilding {n_event_vars} event variables (log(1+x); all-lender suffix '_a') …")
    event_cols = []
    event_data = {}
    for window in WINDOWS:
        for prefix, stem, grp_root, event_label in EVENTS:
            exp = build_experience_one(smp, stem, grp_root, LENDER_SUFFIX, window)
            for dim in DIMS:
                name = f"{window}mo_{event_label}_{dim}"
                event_data[name] = exp[dim].to_numpy(dtype=float)
                event_cols.append(name)
    event_df = pd.DataFrame(event_data, index=smp.index)

    # ── RQ1 columns: 5 DVs + 5 determinants + 21 controls ─────────────────────
    dv_names = [d for d, _ in DV_SPECS]
    rq1_cols = dv_names + DETERMINANTS + CONTROLS
    rq1_df   = smp[rq1_cols].astype(float)
    print(f"RQ1 columns: {len(rq1_cols)} ({len(dv_names)} DVs + {len(DETERMINANTS)} "
          f"determinants + {len(CONTROLS)} controls)")

    # ── Compute both correlation matrices ─────────────────────────────────────
    print(f"\nComputing {len(event_cols)} × {len(rq1_cols)} = {len(event_cols) * len(rq1_cols):,} "
          f"correlations (Pearson + Spearman) …")
    pearson_fmt  = pd.DataFrame("",     index=event_cols, columns=rq1_cols, dtype=object)
    spearman_fmt = pd.DataFrame("",     index=event_cols, columns=rq1_cols, dtype=object)
    pearson_num  = pd.DataFrame(np.nan, index=event_cols, columns=rq1_cols, dtype=float)
    spearman_num = pd.DataFrame(np.nan, index=event_cols, columns=rq1_cols, dtype=float)

    for ev in event_cols:
        x = event_df[ev].to_numpy(dtype=float)
        for rq in rq1_cols:
            y = rq1_df[rq].to_numpy(dtype=float)
            rp, pp = _corr_p(pearsonr,  x, y)
            rs, ps = _corr_p(spearmanr, x, y)
            pearson_fmt.at[ev, rq]  = _fmt_corr(rp, pp)
            spearman_fmt.at[ev, rq] = _fmt_corr(rs, ps)
            pearson_num.at[ev, rq]  = rp
            spearman_num.at[ev, rq] = rs

    pearson_fmt.index.name  = "event var"
    spearman_fmt.index.name = "event var"

    # ── Write xlsx ────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header = ("RQ2 event variables × RQ1 variables. "
              f"Sample: {len(smp):,} obs (RQ2 Model 7 base). "
              "Event vars log(1+x); continuous RQ1 vars winsorized 1%/99%. "
              "Stars: *** p<0.01, ** p<0.05, * p<0.10.")
    with pd.ExcelWriter(OUT_FILE2, engine="xlsxwriter") as xw:
        for sheet, mat in [("pearson", pearson_fmt), ("spearman", spearman_fmt)]:
            mat.to_excel(xw, sheet_name=sheet, startrow=2)
            ws = xw.sheets[sheet]
            ws.write(0, 0, f"{sheet.title()} — {header}")
            ws.freeze_panes(3, 1)

    print(f"\nSaved → {OUT_FILE2}  (sheets: pearson, spearman)")

    # ── Print the top correlates for the robust 12mo cells ────────────────────
    robust_cells = [
        "12mo_IC_Top5_Related",
        "12mo_LF_Top5_Related",
        "12mo_MI_Top5_Related",
    ]
    print(f"\n{'=' * 100}")
    print(f"  Top-10 |correlates| for the {len(robust_cells)} robust RQ2 cells (12mo Top5_Related)")
    print(f"{'=' * 100}")
    for cell in robust_cells:
        print(f"\n{cell}:")
        top = pearson_num.loc[cell].abs().sort_values(ascending=False).head(10).index
        print(f"  {'variable':<22}  {'pearson':>10}  {'spearman':>10}")
        for var in top:
            rp = pearson_num.at[cell, var]
            rs = spearman_num.at[cell, var]
            print(f"  {var:<22}  {rp:>+10.3f}  {rs:>+10.3f}")

    # Small pivot: correlations of the 4 robust cells with the 5 DVs.
    print(f"\n{'=' * 100}")
    print("  Correlation of each robust cell with the 5 DVs (Pearson)")
    print(f"{'=' * 100}")
    dv_slice = pearson_num.loc[robust_cells, dv_names]
    with pd.option_context("display.width", 160,
                           "display.float_format", "{:+.4f}".format):
        print(dv_slice.to_string())


# ── Diagnostic 4: GenAI score decomposition — MNL, no fixed effects ───────────────
# 4.1  MNL of each off-BS score component (SLB/SYN/OPL/VAR-RES) on the RQ1 Model-7
#      covariates with COMPOSITE accounting_policy.
# 4.2  Same, but accounting_policy DECOMPOSED into four dummies — GAAP_OVERRIDE=1/=2,
#      FREEZE=1/=2; joint benchmark = GAAP_OVERRIDE=0 & FREEZE=0.
# The ordinal claude score is an UNORDERED multinomial outcome (baseline level 0). SEs
# clustered by gvkey. NO fixed effects (per instruction). Sample = RQ1 Model-7 (N=10,794).
# Graceful degradation for low-frequency cells: OPL=3 (5 obs) is dropped from the OPL outcome;
# if the rare FREEZE=2 dummy (69 obs) quasi-separates a 4.2 model, it is dropped from that
# model and flagged. The distribution sheet documents both.
DIAG4_OUT = OUT_DIR / "diagnostic4_genai_decomp.xlsx"

# (label, outcome builder, outcome levels dropped from estimation)
SCORE_SPECS = [
    ("SLB",     lambda d: d["claude_SLB_SCORE"],                                   []),
    ("SYN",     lambda d: d["claude_SYN_SCORE"],                                   []),
    ("OPL",     lambda d: d["claude_OPL_SCORE"],                                   [3]),
    ("VAR-RES", lambda d: d[["claude_VAR_SCORE", "claude_RES_SCORE"]].max(axis=1), []),
]
# 4.2 decomposition dummies: (column name, source score, level)
DECOMP = [
    ("GAAP_OVERRIDE_1", "claude_GAAP_OVERRIDE_SCORE", 1),
    ("GAAP_OVERRIDE_2", "claude_GAAP_OVERRIDE_SCORE", 2),
    ("FREEZE_1",        "claude_FREEZE_SCORE",        1),
    ("FREEZE_2",        "claude_FREEZE_SCORE",        2),
]
DECOMP_DET = ([c for c, _, _ in DECOMP] + ["offbslease"] + RATING_BUCKETS
              + ["relationship_freq"])


def _d4_stars(p):
    if pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _d4_fit(y, X, groups):
    """Fit MNLogit; prefer clustered SEs, fall back to plain. Returns (res, cov, converged)."""
    model = sm.MNLogit(y, X)
    attempts = [
        (dict(method="newton", maxiter=200, disp=0,
              cov_type="cluster", cov_kwds={"groups": groups}), "cluster(gvkey)"),
        (dict(method="bfgs", maxiter=3000, disp=0,
              cov_type="cluster", cov_kwds={"groups": groups}), "cluster(gvkey)"),
        (dict(method="bfgs", maxiter=3000, disp=0), "non-clustered (cluster fit failed)"),
    ]
    for kw, cov in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = model.fit(**kw)
            return res, cov, bool(res.mle_retvals.get("converged", False))
        except Exception:                           # noqa: BLE001 — try next method
            continue
    raise RuntimeError("MNLogit failed on all attempts")


def _d4_reliable(res, converged):
    return (converged and np.isfinite(res.prsquared)
            and np.isfinite(np.asarray(res.params, dtype=float)).all()
            and np.isfinite(np.asarray(res.bse, dtype=float)).all())


def _d4_coef_table(res, y):
    nonbase = sorted(pd.unique(y.astype(int)))[1:]           # categories excluding baseline (min)
    P, Pv, Z = res.params, res.pvalues, res.tvalues
    data = {}
    for j, cat in enumerate(nonbase):
        data[(f"y={cat} vs 0", "coef")] = [f"{v:.3f}{_d4_stars(pp)}"
                                            for v, pp in zip(P.iloc[:, j], Pv.iloc[:, j])]
        data[(f"y={cat} vs 0", "z")]    = [f"({zz:.2f})" for zz in Z.iloc[:, j]]
    t = pd.DataFrame(data, index=P.index)
    t.columns = pd.MultiIndex.from_tuples(t.columns)
    return t


def _d4_run(smp, y_full, drop_levels, det, allow_fr2_drop):
    """Run one MNL spec for one component. Returns (table, meta_lines)."""
    keep = ~y_full.isin(drop_levels)
    y = y_full[keep]
    X = sm.add_constant(smp.loc[keep, det + CONTROLS].astype(float))
    g = smp.loc[keep, "gvkey"]
    res, cov, conv = _d4_fit(y, X, g)
    dropped_fr2 = False
    if not _d4_reliable(res, conv) and allow_fr2_drop and "FREEZE_2" in det:
        det2 = [c for c in det if c != "FREEZE_2"]      # rare FREEZE=2 separated → drop, refit
        X = sm.add_constant(smp.loc[keep, det2 + CONTROLS].astype(float))
        res, cov, conv = _d4_fit(y, X, g)
        dropped_fr2 = True
    reliable = _d4_reliable(res, conv)
    meta = [f"N={int(res.nobs):,}  SE={cov}  Converged={conv}  Reliable={reliable}  "
            f"McFadden pseudo-R2={res.prsquared:.4f}"]
    if drop_levels:
        meta.append(f"Dropped outcome level(s) {drop_levels} "
                    f"({int((~keep).sum())} obs) — quasi-separation.")
    if dropped_fr2:
        meta.append("FREEZE_2 dummy dropped (quasi-separation); its rows fold into the Freeze baseline.")
    if not reliable:
        meta.append("⚠ DEGENERATE — coefficients/SEs unreliable.")
    return _d4_coef_table(res, y), meta


def _d4_rq3_decomp():
    """Diagnostic 4.3 — RQ3 (08's RQ3A) with accounting_policy DECOMPOSED into the four
    GAAP_OVERRIDE/FREEZE dummies, in BOTH the main effect and the post_adoption interaction.
    Reuses 08's own tested machinery (prepare_sample, FE builders, winsorize, rank guard,
    clustered OLS) via a bridge so estimates are identical to the paper's RQ3A. Linear
    probability model, IY+Borrower+Lender FEs, SE clustered by gvkey, rank guard. Sample =
    08's adopting-firm RQ3 sample (N≈5,549 per DV). Returns (table, meta)."""
    path = REPO_DIR / "08_rq3_asc842.py"
    g = {"__name__": "_d4_bridge", "__file__": str(path)}       # exec WITHOUT running 08's main()
    exec(compile(path.read_text(), str(path), "exec"), g)

    lender_lists = g["load_lender_lists"]()
    df, X_fe, fe_bor, fe_lender, _ = g["prepare_sample"](lender_lists)
    for name, src, lvl in DECOMP:
        df[name] = (df[src] == lvl).astype(float)

    POST, iname = g["POST"], g["interaction_name"]
    CTRL8, WINSOR8 = g["CONTROLS"], g["WINSOR_LEVEL_VARS"]
    dec = [c for c, _, _ in DECOMP]
    dec_det = dec + ["offbslease"] + RATING_BUCKETS + ["relationship_freq"]
    dec_int = dec + ["relationship_freq", "fin_covenant_count", "offbslease"] + RATING_BUCKETS + ["amendment"]
    test = [POST] + [iname(v) for v in dec_int]
    regressors = test + dec_det + CTRL8
    show_rows = test + dec_det                                   # rows shown (controls estimated, not shown)

    # RQ3 estimation-sample distribution of the decomposed categories × post_adoption. The mask
    # is DV-invariant (all score dummies share missingness), so one distribution covers all DVs.
    # This documents which categories can be identified under the full FE (the §4.3 bullet): a
    # category needs variation on BOTH sides of post_adoption for its post× interaction to survive.
    any_dv = f"{g['DV_SPECS'][0][0]}_score_dummy"
    est_mask = (pd.Series(np.isfinite(df[[POST] + dec_det + CTRL8].to_numpy(dtype=float)).all(axis=1),
                          index=df.index)
                & df[any_dv].notna() & df["gvkey"].notna())
    est = df[est_mask]
    go_e = est["claude_GAAP_OVERRIDE_SCORE"].astype(int)
    fr_e = est["claude_FREEZE_SCORE"].astype(int)
    rq3_dist = pd.DataFrame(
        [{"category": cn, "total": int(c.sum()),
          "post=0": int((c & (est[POST] == 0)).sum()),
          "post=1": int((c & (est[POST] == 1)).sum())}
         for cn, c in [("GAAP_OVERRIDE=1", go_e == 1), ("GAAP_OVERRIDE=2", go_e == 2),
                       ("FREEZE=1", fr_e == 1), ("FREEZE=2", fr_e == 2),
                       ("benchmark GO=0 & FR=0", (go_e == 0) & (fr_e == 0))]]
    ).set_index("category")
    print(f"  RQ3 estimation sample N={len(est):,} — decomposition × post_adoption:")
    print(rq3_dist.to_string())

    cols, ns = {}, {}
    for sheet, _cols in g["DV_SPECS"]:
        dv = f"{sheet}_score_dummy"
        parents = [POST] + dec_det + CTRL8
        mask = (pd.Series(np.isfinite(df[parents].to_numpy(dtype=float)).all(axis=1), index=df.index)
                & df[dv].notna() & df["gvkey"].notna())
        dfw = g["winsorize_cols"](df, WINSOR8, mask)            # winsorize FIRST, then interactions
        for v in dec_int:
            dfw[iname(v)] = dfw[POST] * dfw[v]
        X_full = pd.concat([dfw[regressors].astype(float), X_fe, fe_bor, fe_lender], axis=1)
        X, y, cl = g["stabilize_design"](X_full, dfw[dv], dfw["gvkey"])
        res = g["fit_ols_clustered"](y, X, cl)
        ns[sheet] = len(y)
        coef, tval = [], []
        for v in show_rows:
            if v in res.params.index:
                coef.append(f"{res.params[v]:.3f}{_d4_stars(res.pvalues[v])}")
                tval.append(f"({res.tvalues[v]:.2f})")
            else:
                coef.append("dropped"); tval.append("")
        cols[(sheet, "coef")] = coef
        cols[(sheet, "t")]    = tval
        print(f"  RQ3 {sheet}: N={len(y):,}  post_adoption={res.params.get(POST, float('nan')):.3f}")

    tbl = pd.DataFrame(cols, index=show_rows)
    tbl.columns = pd.MultiIndex.from_tuples(tbl.columns)
    meta = [
        "Diagnostic 4.3 — RQ3 (08 RQ3A) with accounting_policy decomposed (main effect + post× interaction).",
        "Linear probability model; IY + Borrower + Lender FEs; SE clustered by gvkey; rank guard — identical machinery to 08.",
        f"Estimation N per DV: {ns}.  The 21 controls + amendment are included but not shown.  "
        "'dropped' = removed by the rank guard (collinear/low-frequency under the full FE).",
        "See the 'distributions' sheet for the RQ3-sample category counts × post_adoption "
        "(which decomposed terms are identified under the full FE).",
    ]
    return tbl, meta, rq3_dist


def diagnostic4_genai_decomp():
    """Diagnostic 4.1 + 4.2 (MNL, no FE) and 4.3 (RQ3 linear decomposition) in one workbook."""
    print("Loading contracts …")
    df = pd.read_parquet(DATA_DIR / "contracts.parquet")
    df["tranche_active_date"] = pd.to_datetime(df["tranche_active_date"])
    smp = prepare_sample(df).reset_index(drop=True)
    print(f"RQ1 Model-7 sample: N={len(smp):,}")

    for c in WINSOR_VARS:                                # winsorize on the estimation sample (as in 06)
        lo, hi = smp[c].quantile(0.01), smp[c].quantile(0.99)
        smp[c] = smp[c].clip(lo, hi)
    for name, src, lvl in DECOMP:                        # 4.2 decomposition dummies
        smp[name] = (smp[src] == lvl).astype(float)

    # ── Distributions (sheet 1) ──────────────────────────────────────────────────
    outcome_rows = []
    for label, builder, _ in SCORE_SPECS:
        vc = builder(smp).astype(int).value_counts().sort_index()
        outcome_rows.append({"component": label, **{f"level_{k}": int(v) for k, v in vc.items()}})
    outcome_dist = pd.DataFrame(outcome_rows).set_index("component").fillna(0).astype(int)

    go = smp["claude_GAAP_OVERRIDE_SCORE"].astype(int)
    fr = smp["claude_FREEZE_SCORE"].astype(int)
    decomp_rows = [{"category": name, "n": int((smp[src].astype(int) == lvl).sum()),
                    "pct": round((smp[src].astype(int) == lvl).mean() * 100, 1)}
                   for name, src, lvl in DECOMP]
    decomp_rows.append({"category": "benchmark GO=0 & FR=0",
                        "n": int(((go == 0) & (fr == 0)).sum()),
                        "pct": round(((go == 0) & (fr == 0)).mean() * 100, 1)})
    decomp_dist = pd.DataFrame(decomp_rows).set_index("category")
    crosstab = pd.crosstab(go.rename("GAAP_OVERRIDE"), fr.rename("FREEZE"), margins=True)

    # ── Fit 4.1 (composite) and 4.2 (decomposed) for each component ───────────────
    results = []
    for label, builder, drop_levels in SCORE_SPECS:
        y_full = builder(smp).astype(int)
        print(f"\n{label}: outcome {y_full.value_counts().sort_index().to_dict()}")
        t1, m1 = _d4_run(smp, y_full, drop_levels, DETERMINANTS, allow_fr2_drop=False)
        t2, m2 = _d4_run(smp, y_full, drop_levels, DECOMP_DET, allow_fr2_drop=True)
        results.append((f"4.1_{label}", "4.1 composite accounting_policy", label, t1, m1))
        results.append((f"4.2_{label}", "4.2 decomposed accounting_policy", label, t2, m2))
        print(f"  4.1: {m1[0]}")
        print(f"  4.2: {m2[0]}")

    # ── 4.3: RQ3 linear decomposition (reuses 08's RQ3A machinery) ────────────────
    print("\n── 4.3 RQ3 decomposition (08 RQ3A, accounting_policy decomposed) ──")
    rq3_tbl, rq3_meta, rq3_dist = _d4_rq3_decomp()

    # ── Write the single diagnostic-4 workbook ───────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(DIAG4_OUT, engine="xlsxwriter") as xw:
        outcome_dist.to_excel(xw, sheet_name="distributions", startrow=1)
        r = len(outcome_dist) + 4
        decomp_dist.to_excel(xw, sheet_name="distributions", startrow=r)
        r2 = r + len(decomp_dist) + 3
        crosstab.to_excel(xw, sheet_name="distributions", startrow=r2)
        ws = xw.sheets["distributions"]
        ws.write(0, 0, "Diagnostic 4 — distributions (RQ1 Model-7 sample, N=10,794)")
        ws.write(len(outcome_dist) + 3, 0,
                 "4.2 decomposition categories (dummies NOT mutually exclusive; benchmark = GO=0 & FR=0):")
        ws.write(r2 - 1, 0, "Cross-tab: GAAP_OVERRIDE (rows) x FREEZE (cols):")
        r3 = r2 + len(crosstab) + 3
        ws.write(r3 - 1, 0,
                 "4.3 RQ3 estimation-sample decomposition counts x post_adoption "
                 "(governs which decomposed terms are identified under the full FE):")
        rq3_dist.to_excel(xw, sheet_name="distributions", startrow=r3)
        for sheet, title, label, tbl, meta in results:
            head = [f"Diagnostic {title} — MNL: {label} score, no FE (multinomial, baseline = level 0)"] + meta
            tbl.to_excel(xw, sheet_name=sheet, startrow=len(head) + 1)
            wsx = xw.sheets[sheet]
            for i, line in enumerate(head):
                wsx.write(i, 0, line)

        # 4.3 RQ3 decomposition — one sheet, five DV columns
        rq3_tbl.to_excel(xw, sheet_name="4.3_RQ3_decomp", startrow=len(rq3_meta) + 1)
        wsx = xw.sheets["4.3_RQ3_decomp"]
        for i, line in enumerate(rq3_meta):
            wsx.write(i, 0, line)

    print(f"\nSaved → {DIAG4_OUT}  (sheets: distributions, "
          f"{', '.join(s for s, *_ in results)}, 4.3_RQ3_decomp)")


# ── Diagnostic 5: RQ3A robustness — non-adopters as never-treated controls ────────
# Re-estimates 08's RQ3A with NON-ADOPTERS KEPT and coded post_adoption = 0 (a never-treated
# control group). 08's base code is NOT modified — this only bridges to 08's tested machinery and
# reuses 08's build_rq3a(), so the output sheet is in the IDENTICAL format to 08's RQ3A. Composite
# accounting_policy (matches 08's base RQ3A, so the headline post × accounting_policy is directly
# comparable). Only the never-adopter sheet is produced — the adopters-only column is 08's own output
# (rq3_asc842*.xlsx). The ONLY change vs 08 is the non-adopter post_adoption coding (0 not NaN).
DIAG5_OUT = OUT_DIR / "diagnostic5_rq3_fullsample.xlsx"


def _d5_build_neveradopt(g, lender_lists):
    """08.prepare_sample replicated, but non-adopters are KEPT with post_adoption = 0 (never-treated
    controls) instead of dropped. Returns (df, X_fe, fe_bor, fe_lender, fe_labels, adopt) — matching
    08.prepare_sample's return, plus the adopt series for the composition note."""
    df = pd.read_parquet(g["CONTRACTS"])
    df["tranche_active_date"] = pd.to_datetime(df["tranche_active_date"])
    df = df[df["claude_is_debt_contract"] == "Y"].copy()
    df = df[df[g["DV_SCORES"]].notna().all(axis=1)].copy()
    for sheet, cols in g["DV_SPECS"]:
        df[f"{sheet}_score_dummy"] = (df[cols] > 0).any(axis=1).astype(float)
    df["contract_year"] = df["tranche_active_date"].dt.year
    gaap, freeze = df["claude_GAAP_OVERRIDE_SCORE"], df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan
    df[g["COVENANT_RATIO_FILL"]] = df[g["COVENANT_RATIO_FILL"]].fillna(0)

    adopt = pd.to_datetime(df["adoption_date"], errors="coerce")
    # NEVER-ADOPTER coding: non-adopters → 0 (not NaN); no adopter restriction.
    df[g["POST"]] = np.where(adopt.isna(), 0.0,
                             (df["tranche_active_date"] >= adopt).astype(float))

    sic = 2
    fe_iy, n_cells = g["make_industry_year_fe"](df, sic)
    while n_cells >= len(df) - 5 and sic > 1:
        sic -= 1
        fe_iy, n_cells = g["make_industry_year_fe"](df, sic)
    X_fe = fe_iy.loc[:, (fe_iy != 0).any(axis=0)]
    fe_bor = g["make_borrower_fe"](df)
    df = df.join(lender_lists, on=g["MERGE_KEYS"])
    df["lender_ids"] = df["lender_ids"].apply(lambda x: x if isinstance(x, list) else [])
    fe_lender = g["make_lender_multi_hot"](df, df["lender_ids"])
    fe_labels = [f"Industry×Year FEs (SIC {sic}-digit)", "Borrower FEs", "Lender FEs"]
    return df, X_fe, fe_bor, fe_lender, fe_labels, adopt


def diagnostic5_never_adopter():
    """Diagnostic 5 — RQ3A robustness with non-adopters as never-treated controls (post_adoption=0).
    Produces ONLY the never-adopter sheet, via 08's own build_rq3a(), so it is in the IDENTICAL format
    to 08's RQ3A sheet — full regressor + control rows and the N / R² / Adj. R² / FE-count footer.
    The adopters-only column is NOT regenerated here (it is 08's own RQ3A output, rq3_asc842*.xlsx);
    compare `RQ3A_never_adopter` line-by-line against that. 08's base code is UNCHANGED. Slow (~4 min:
    full FE on the ~10,794 never-adopter sample). Writes diagnostic5_rq3_fullsample.xlsx."""
    path = REPO_DIR / "08_rq3_asc842.py"
    g = {"__name__": "_d5_bridge", "__file__": str(path)}       # exec WITHOUT running 08's main()
    exec(compile(path.read_text(), str(path), "exec"), g)

    lender_lists = g["load_lender_lists"]()
    print("\n[never-adopter] non-adopters kept as post_adoption = 0 controls …")
    dfB, XfeB, borB, lenB, labelsB, adoptB = _d5_build_neveradopt(g, lender_lists)
    sheet_B = g["build_rq3a"](dfB, XfeB, borB, lenB, labelsB)

    POST = g["POST"]
    n_post1       = int((dfB[POST] == 1).sum())
    n_post0_adopt = int(((dfB[POST] == 0) & adoptB.notna()).sum())
    n_post0_never = int(((dfB[POST] == 0) & adoptB.isna()).sum())
    notes = [
        "Diagnostic 5 — RQ3A robustness: non-adopters KEPT as never-treated controls (post_adoption = 0).",
        "Same spec as 08's RQ3A, produced by 08's own build_rq3a() → IDENTICAL format (compare line by line).",
        "The adopters-only column is 08's own output (rq3_asc842*.xlsx 'RQ3A') — NOT regenerated here.",
        "08's base code is UNCHANGED (bridge-reused). Composite accounting_policy; IY+Borrower+Lender FEs;",
        "SE clustered by gvkey; rank guard. Columns (1)-(5) = SLB, SYN, OPL, VAR-RES, ALL score dummies.",
        f"Never-adopter sample ({len(dfB):,} rows pre-mask): post=1 {n_post1:,}; "
        f"post=0 pre-adoption {n_post0_adopt:,}; post=0 never-adopters {n_post0_never:,}.",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(DIAG5_OUT, engine="xlsxwriter") as xw:
        ws_notes = xw.book.add_worksheet("notes")
        for i, line in enumerate(notes):
            ws_notes.write(i, 0, line)
        sheet_B.to_excel(xw, sheet_name="RQ3A_never_adopter")

    print(f"\nSaved → {DIAG5_OUT}  (sheets: notes, RQ3A_never_adopter)")


if __name__ == "__main__":
    rq2_distributions()
    print("\n\n")
    rq2_rq1_correlations()
    print("\n\n")
    diagnostic4_genai_decomp()
    print("\n\n")
    diagnostic5_never_adopter()
