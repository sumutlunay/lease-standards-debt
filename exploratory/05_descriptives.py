"""
05_descriptives.py
Descriptive statistics + correlation matrix for the variables used in
06_rq1_determinants.py. Two worksheets in one workbook.

Runs BEFORE the regression (06) in the pipeline as a stylistic preference (descriptives
indexed ahead of the models), but still describes exactly the regression's estimation
sample: it re-derives the identical Models 7/8 sample mask from fulldata.parquet
independently, so no dependency on 06 having run first.

SHEET 1 "descriptives" — N, min, p10, p25, p50, p75, p90, max, mean, std per variable.
Two presentation rules, per Sunay:
  • LOGGED variables are reported in their NON-LOGGED (level) form. Every logged
    variable upstream is a plain np.log(x) (with x>0 masking), so exp() recovers
    the exact level. These levels are NOT winsorized (the log already tames skew).
  • Other CONTINUOUS variables are reported as the WINSORIZED versions that enter
    the regression — 1%/99% both tails, on the estimation sample (same as 06's
    Models 7/8: WINSOR_LEVEL_VARS).
  • Dummies / integer scales / fractions (the five DV dummies, accounting_policy,
    the credit-quality bucket dummies, secured, perf_pricing, loss,
    relationship_freq) are reported as they enter — no transform.
  • The two COMPONENTS of accounting_policy — gaap_override (claude_GAAP_OVERRIDE_SCORE > 0)
    and freeze (claude_FREEZE_SCORE > 0) — are reported as descriptive dummies directly under
    accounting_policy. They are NOT regressors (accounting_policy = the OR of the two enters the
    models), so they appear on the descriptives sheet only, not in the correlation matrix.
  • CREDIT QUALITY enters as mutually-exclusive BUCKET dummies (BB_grade, B_grade,
    CCC_below, non_rated_suppl_all) rather than the linear num_rating_suppl_all scale.
    ig_grade (investment grade, BBB- or above) is the OMITTED reference category: it is
    reported on the descriptives sheet for completeness but is NOT a regressor, so — like
    the accounting_policy components — it is excluded from the correlation matrix.

SHEET 2 "correlations" — correlation matrix of all regression variables IN
REGRESSION FORM (log transforms and winsorization LEFT IN — not undone). Upper
triangle = Spearman, lower triangle = Pearson, diagonal = 1.00. Coefficients to
2 decimals with significance stars: * p<0.10, ** p<0.05, *** p<0.01.

DVs (each a dummy = 1 if ANY listed claude_*_SCORE > 0, else 0; see DV_SPECS):
  SLB, SYN, OPL (single score); VAR-RES (VAR or RES); ALL (any of the five).

Sample (both sheets): the full-model estimation sample used by Models 7–8 in 06 —
  claude_is_debt_contract == "Y", all 5 determinants and 21 controls non-missing,
  every claude_*_SCORE non-missing, gvkey non-missing. Winsorization bounds are
  computed on THIS sample, so they reproduce the exact bounds used in Models 7/8.
  (To use a broader sample, relax the mask below.)

Input:  data/fulldata.parquet    (output of 03_contracts.py + 04_merge.py)
Output: output/tables/descriptives.xlsx  (sheets: descriptives, correlations)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_DIR = Path(__file__).resolve().parent.parent  # code/ — this script lives in code/exploratory/
DATA_DIR = REPO_DIR.parent / "data"
OUT_DIR  = REPO_DIR.parent / "output" / "tables"
OUT_FILE = OUT_DIR / "descriptives.xlsx"

# DV dummies (1 if ANY listed claude_*_SCORE > 0, else 0) — same construction as 05.
DV_SPECS = [
    ("SLB_score_dummy",     ["claude_SLB_SCORE"]),
    ("SYN_score_dummy",     ["claude_SYN_SCORE"]),
    ("OPL_score_dummy",     ["claude_OPL_SCORE"]),
    ("VAR-RES_score_dummy", ["claude_VAR_SCORE", "claude_RES_SCORE"]),
    ("ALL_score_dummy",     ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
                             "claude_VAR_SCORE", "claude_RES_SCORE"]),
]

# Kept in lock-step with 06_rq1_determinants.py (DETERMINANTS / CONTROLS / WINSOR_LEVEL_VARS).
DETERMINANTS = ["accounting_policy", "offbslease", "BB_grade", "B_grade", "CCC_below", "non_rated_suppl_all", "relationship_freq"]
CONTROLS = [
    "maturity", "log_lender_count", "log_interest", "log_deal_amount", "perf_pricing",
    "fin_covenant_count", "gen_covenant_count", "secured",
    "size", "profitability", "bsfixed", "liabilities", "logage", "btm", "capex",
    "loss", "rand", "divyield",
    "log_bond_count", "bond_proceeds_scaled",
]

# Continuous vars winsorized at 1%/99% in 06 (== WINSOR_LEVEL_VARS there).
WINSOR_VARS = [
    "offbslease",
    "fin_covenant_count", "gen_covenant_count",
    "profitability", "bsfixed", "liabilities", "btm", "capex", "rand", "divyield",
    "bond_proceeds_scaled",
]

# Output row plan: (regression var, display label, kind).
#   kind: "dummy" (0/1), "asis" (no transform), "winsor" (1%/99%), "log" (report exp()),
#         "log1p" (report exp()-1, for log(1+x) vars such as logage).
PLAN = [
    # Dependent variables
    ("SLB_score_dummy",     "SLB_score_dummy  (DV)",                      "dummy"),
    ("SYN_score_dummy",     "SYN_score_dummy  (DV)",                      "dummy"),
    ("OPL_score_dummy",     "OPL_score_dummy  (DV)",                      "dummy"),
    ("VAR-RES_score_dummy", "VAR-RES_score_dummy  (DV)",                  "dummy"),
    ("ALL_score_dummy",     "ALL_score_dummy  (DV)",                      "dummy"),
    # Determinants
    ("accounting_policy", "accounting_policy",                            "dummy"),
    ("gaap_override",     "  gaap_override  (accounting_policy component)", "dummy"),
    ("freeze",            "  freeze  (accounting_policy component)",        "dummy"),
    ("offbslease",        "offbslease",                                   "winsor"),
    ("ig_grade",          "  ig_grade  (OMITTED rating reference)",       "dummy"),
    ("BB_grade",          "BB_grade",                                     "dummy"),
    ("B_grade",           "B_grade",                                      "dummy"),
    ("CCC_below",         "CCC_below",                                    "dummy"),
    ("non_rated_suppl_all",  "non_rated_suppl_all",                        "dummy"),
    ("relationship_freq", "relationship_freq",                            "asis"),
    # Deal-level controls
    ("maturity",          "maturity_months  (=exp(maturity))",            "log"),
    ("log_lender_count",  "number_of_lenders  (=exp(log_lender_count))",  "log"),
    ("log_interest",      "all_in_spread_bps  (=exp(log_interest))",      "log"),
    ("log_deal_amount",   "deal_amount_musd  (=exp(log_deal_amount))",    "log"),
    ("perf_pricing",      "perf_pricing",                                 "dummy"),
    ("fin_covenant_count","fin_covenant_count",                           "winsor"),
    ("gen_covenant_count","gen_covenant_count",                           "winsor"),
    ("secured",           "secured",                                      "dummy"),
    # Firm-level controls
    ("size",              "total_assets_musd  (=exp(size))",              "log"),
    ("profitability",     "profitability",                                "winsor"),
    ("bsfixed",           "bsfixed",                                      "winsor"),
    ("liabilities",       "liabilities",                                  "winsor"),
    ("logage",            "firm_age_years  (=exp(logage)-1)",             "log1p"),
    ("btm",               "btm",                                          "winsor"),
    ("capex",             "capex",                                        "winsor"),
    ("loss",              "loss",                                         "dummy"),
    ("rand",              "rand",                                         "winsor"),
    ("divyield",          "divyield",                                     "winsor"),
    # FISD bond-activity controls (added with 06's control set)
    ("log_bond_count",       "bond_issuance_count  (=exp(log_bond_count)-1)", "log1p"),
    ("bond_proceeds_scaled", "bond_proceeds_scaled",                          "winsor"),
]

TRANSFORM_NOTE = {
    "dummy":  "indicator (0/1)",
    "asis":   "as entered (no transform)",
    "winsor": "winsorized 1%/99%",
    "log":    "non-logged level = exp(.)",
    "log1p":  "non-logged level = exp(.) - 1",
}

STAT_COLS = ["min", "p10", "p25", "p50", "p75", "p90", "max", "mean", "std"]


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


# ── Correlation matrix helpers ─────────────────────────────────────────────────────

def _corr_star(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def _corr_p(func, x: np.ndarray, y: np.ndarray):
    """Return (coef, p-value) from pearsonr/spearmanr across scipy versions
    (newer return a result object with .statistic/.pvalue; older return a tuple)."""
    res = func(x, y)
    try:
        return float(res.statistic), float(res.pvalue)
    except AttributeError:
        return float(res[0]), float(res[1])


def _fmt_corr(r: float, p: float) -> str:
    s = f"{r:.2f}"
    if s == "-0.00":
        s = "0.00"
    return s + _corr_star(p)


def build_corr_matrix(data: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Square correlation matrix. Upper triangle = Spearman, lower = Pearson,
    diagonal = 1.00. Cells are formatted strings (coef to 2dp + significance stars).
    Variables enter in REGRESSION FORM (logs kept, winsorization kept)."""
    arrs = {v: data[v].to_numpy(dtype=float) for v in variables}
    M = pd.DataFrame("", index=variables, columns=variables, dtype=object)
    for i, vi in enumerate(variables):
        for j, vj in enumerate(variables):
            if i == j:
                M.iat[i, j] = "1.00"
            elif i < j:                          # upper triangle → Spearman
                r, p = _corr_p(spearmanr, arrs[vi], arrs[vj])
                M.iat[i, j] = _fmt_corr(r, p)
            else:                                # lower triangle → Pearson
                r, p = _corr_p(pearsonr, arrs[vi], arrs[vj])
                M.iat[i, j] = _fmt_corr(r, p)
    M.index.name = "upper=Spearman / lower=Pearson"
    return M


def run():
    print("Loading contracts …")
    df = pd.read_parquet(DATA_DIR / "fulldata.parquet")
    print(f"  {len(df):,} rows × {df.shape[1]} cols")

    # ── Reconstruct the regression variables exactly as 05 does ────────────────
    df = df[df["claude_is_debt_contract"] == "Y"].copy()
    print(f"  {len(df):,} rows after keeping claude_is_debt_contract == Y")

    for dv, score_cols in DV_SPECS:
        df[dv] = (df[score_cols] > 0).any(axis=1).astype(float)

    # accounting_policy: OR logic; missing kept as missing (matches 05).
    gaap   = df["claude_GAAP_OVERRIDE_SCORE"]
    freeze = df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan

    # The two components of accounting_policy, reported for descriptives only (not regressors).
    # Same missing-kept-as-missing rule, applied per component (NaN where its own source score is).
    df["gaap_override"] = (gaap > 0).astype(float)
    df.loc[gaap.isna(), "gaap_override"] = np.nan
    df["freeze"] = (freeze > 0).astype(float)
    df.loc[freeze.isna(), "freeze"] = np.nan

    # ── Estimation sample (matches Models 7/8) ─────────────────────────────────
    need_cols = DETERMINANTS + CONTROLS
    finite    = np.isfinite(df[need_cols].to_numpy(dtype=float)).all(axis=1)
    all_score_cols = sorted({c for _, cols in DV_SPECS for c in cols})
    score_ok  = df[all_score_cols].notna().all(axis=1)
    mask      = pd.Series(finite, index=df.index) & score_ok & df["gvkey"].notna()
    smp = df[mask].copy()
    print(f"  {len(smp):,} rows in the full-model estimation sample "
          f"(all determinants + controls non-missing)")

    # ── Winsorize the continuous vars at 1%/99% on this sample ─────────────────
    print("\nWinsorizing continuous variables at 1%/99% on the estimation sample:")
    for c in WINSOR_VARS:
        lo, hi = smp[c].quantile(0.01), smp[c].quantile(0.99)
        n_clip = int(((smp[c] < lo) | (smp[c] > hi)).sum())
        smp[c] = smp[c].clip(lower=lo, upper=hi)
        print(f"    {c:<20} [{lo:.4f}, {hi:.4f}]  ({n_clip} obs clipped)")

    # ── Build each reported series and compute stats ───────────────────────────
    rows = {}
    order = []
    for reg_var, label, kind in PLAN:
        if kind == "log":
            series = np.exp(smp[reg_var])          # non-logged level (unwinsorized)
        elif kind == "log1p":
            series = np.expm1(smp[reg_var])        # log(1+x) var -> true level = exp() - 1
        else:
            series = smp[reg_var]                  # dummy / asis / already-winsorized
        stats = compute_stats(series)
        stats = {"variable": reg_var, "transform": TRANSFORM_NOTE[kind], **stats}
        rows[label] = stats
        order.append(label)

    out = pd.DataFrame(rows).T.loc[order]
    out = out[["variable", "transform", "N"] + STAT_COLS]
    # Round only the numeric stat columns for display.
    out[["N"] + STAT_COLS] = out[["N"] + STAT_COLS].apply(pd.to_numeric)
    out[STAT_COLS] = out[STAT_COLS].round(4)
    out["N"] = out["N"].astype(int)

    # ── Correlation matrix (regression form: logs + winsorization kept) ────────
    dv_names  = [dv for dv, _ in DV_SPECS]
    corr_vars = dv_names + DETERMINANTS + CONTROLS
    print(f"\nBuilding correlation matrix over {len(corr_vars)} regression variables "
          f"(upper=Spearman, lower=Pearson) …")
    corr = build_corr_matrix(smp, corr_vars)

    OUT_DIR.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        out.to_excel(xw,  sheet_name="descriptives", index=True)
        corr.to_excel(xw, sheet_name="correlations", index=True)

    print(f"\nSaved → {OUT_FILE}  (sheets: descriptives, correlations)")
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print("\n" + out.to_string())


if __name__ == "__main__":
    run()
