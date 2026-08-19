"""
table1A_descriptives.py  —  Table 1, Panel A: descriptive statistics
====================================================================

Manuscript Table 1 Panel A. Reports Mean, P25, P50, P75 and Std for every variable
entering the analyses, on the Model 7 estimation sample (N = 11,184).

Leaner than the exploratory 05_descriptives.py in three ways:
  • five statistics only (05 also reports N/min/p10/p90/max),
  • no correlation matrix (05's second sheet),
  • no display relabeling — rows carry the ORIGINAL variable names.

Row order follows the reference table (output/tables/Tables 08-13-26.xlsx, sheet
"Table1"), which differs from 05's PLAN in two places: the overall recognition dummy
(ALL) leads the DV block rather than trailing it, and `secured` sits immediately after
`relationship_freq` rather than after the covenant counts.

★ RATING SPEC. The reference table's single linear "Credit rating" row
(num_rating_suppl_all) is RETIRED and replaced by the credit-quality buckets, matching
the regression spec adopted across the pipeline in 7985df5. `ig_grade` is the OMITTED
REFERENCE in the regressions and is therefore not a regressor, but it IS reported here
so the five categories visibly sum to 1.0 and readers can size the base category when
interpreting the bucket coefficients. Drop it from ROW_PLAN if the table should show
estimated variables only.

Variable transforms follow 05 exactly, so the figures reproduce the reference table:
logged regressors are reported as non-logged levels via exp() (or exp()-1 for the two
log(1+x) variables), and the continuous level variables are winsorized 1%/99% on the
estimation sample.

Input : data/fulldata.parquet
Output: output/tables/table1A.xlsx
"""

import numpy as np
import pandas as pd
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent   # code/ — this script lives in code/manuscript/
DATA_DIR = REPO_DIR.parent / "data"
OUT_DIR  = REPO_DIR.parent / "output" / "tables"
OUT_FILE = OUT_DIR / "table1A.xlsx"

# ── Regression spec — kept in lock-step with 06_rq1_determinants.py ──────────────
DETERMINANTS = ["accounting_policy", "offbslease", "BB_grade", "B_grade", "CCC_below",
                "non_rated_suppl_all", "relationship_freq"]
CONTROLS = [
    "maturity", "log_lender_count", "log_interest", "log_deal_amount", "perf_pricing",
    "fin_covenant_count", "gen_covenant_count", "secured",
    "size", "profitability", "bsfixed", "liabilities", "logage", "btm", "capex",
    "loss", "rand", "divyield",
    "log_bond_count", "bond_proceeds_scaled",
]

DV_SPECS = [
    ("SLB_score_dummy",     ["claude_SLB_SCORE"]),
    ("SYN_score_dummy",     ["claude_SYN_SCORE"]),
    ("OPL_score_dummy",     ["claude_OPL_SCORE"]),
    ("VAR-RES_score_dummy", ["claude_VAR_SCORE", "claude_RES_SCORE"]),
    ("ALL_score_dummy",     ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
                             "claude_VAR_SCORE", "claude_RES_SCORE"]),
]

# Continuous level variables winsorized 1%/99% in 06 (== 06's WINSOR_LEVEL_VARS).
WINSOR_VARS = [
    "offbslease",
    "fin_covenant_count", "gen_covenant_count",
    "profitability", "bsfixed", "liabilities", "btm", "capex", "rand", "divyield",
    "bond_proceeds_scaled",
]

# ── Row plan: (variable, transform kind). Order = the reference table's order. ───
#   kind: "asis"   report as it enters (dummies, fractions, already-winsorized levels)
#         "log"    logged regressor  → report exp(.)
#         "log1p"  log(1+x) regressor → report exp(.) - 1
ROW_PLAN = [
    # Recognition dummies (DVs) — overall first, per the reference table
    ("ALL_score_dummy",     "asis"),
    ("SLB_score_dummy",     "asis"),
    ("SYN_score_dummy",     "asis"),
    ("OPL_score_dummy",     "asis"),
    ("VAR-RES_score_dummy", "asis"),
    # Determinants
    ("accounting_policy",   "asis"),
    ("gaap_override",       "asis"),   # component of accounting_policy, descriptive only
    ("freeze",              "asis"),   # component of accounting_policy, descriptive only
    ("offbslease",          "asis"),
    # Credit-quality buckets (replacing the reference table's linear "Credit rating")
    ("ig_grade",            "asis"),   # OMITTED REFERENCE in the regressions
    ("BB_grade",            "asis"),
    ("B_grade",             "asis"),
    ("CCC_below",           "asis"),
    ("non_rated_suppl_all", "asis"),
    ("relationship_freq",   "asis"),
    # Deal-level controls
    ("secured",             "asis"),
    ("maturity",            "log"),
    ("log_lender_count",    "log"),
    ("log_interest",        "log"),
    ("log_deal_amount",     "log"),
    ("perf_pricing",        "asis"),
    ("fin_covenant_count",  "asis"),
    ("gen_covenant_count",  "asis"),
    # Firm-level controls
    ("size",                "log"),
    ("profitability",       "asis"),
    ("bsfixed",             "asis"),
    ("liabilities",         "asis"),
    ("logage",              "log1p"),
    ("btm",                 "asis"),
    ("capex",               "asis"),
    ("loss",                "asis"),
    ("rand",                "asis"),
    ("divyield",            "asis"),
    # FISD bond-activity controls
    ("log_bond_count",      "log1p"),
    ("bond_proceeds_scaled","asis"),
]

STAT_COLS = ["Mean", "P25", "P50", "P75", "Std"]


def compute_stats(s: pd.Series) -> dict:
    s = s.astype(float)
    return {
        "Mean": s.mean(),
        "P25":  s.quantile(0.25),
        "P50":  s.quantile(0.50),
        "P75":  s.quantile(0.75),
        "Std":  s.std(ddof=1),
    }


def build_sample() -> pd.DataFrame:
    """The Model 7 estimation sample, re-derived exactly as 05/06 derive it."""
    print("Loading contracts …")
    df = pd.read_parquet(DATA_DIR / "fulldata.parquet")
    print(f"  {len(df):,} rows × {df.shape[1]} cols")

    df = df[df["claude_is_debt_contract"] == "Y"].copy()
    print(f"  {len(df):,} rows after keeping claude_is_debt_contract == Y")

    for dv, score_cols in DV_SPECS:
        df[dv] = (df[score_cols] > 0).any(axis=1).astype(float)

    # accounting_policy: OR logic, missing kept as missing.
    gaap, freeze = df["claude_GAAP_OVERRIDE_SCORE"], df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan
    # Its two components — reported descriptively, never regressors.
    df["gaap_override"] = (gaap > 0).astype(float)
    df.loc[gaap.isna(), "gaap_override"] = np.nan
    df["freeze"] = (freeze > 0).astype(float)
    df.loc[freeze.isna(), "freeze"] = np.nan

    finite   = np.isfinite(df[DETERMINANTS + CONTROLS].to_numpy(dtype=float)).all(axis=1)
    score_ok = df[sorted({c for _, cols in DV_SPECS for c in cols})].notna().all(axis=1)
    mask     = pd.Series(finite, index=df.index) & score_ok & df["gvkey"].notna()
    smp      = df[mask].copy()
    print(f"  {len(smp):,} rows in the Model 7 estimation sample")

    print("\nWinsorizing continuous variables at 1%/99% on the estimation sample:")
    for c in WINSOR_VARS:
        lo, hi = smp[c].quantile(0.01), smp[c].quantile(0.99)
        n_clip = int(((smp[c] < lo) | (smp[c] > hi)).sum())
        smp[c] = smp[c].clip(lower=lo, upper=hi)
        print(f"    {c:<22} [{lo:.4f}, {hi:.4f}]  ({n_clip} obs clipped)")
    return smp


def run() -> None:
    smp = build_sample()

    rows = {}
    for var, kind in ROW_PLAN:
        if kind == "log":
            series = np.exp(smp[var])          # non-logged level
        elif kind == "log1p":
            series = np.expm1(smp[var])        # log(1+x) → true level
        else:
            series = smp[var]
        rows[var] = compute_stats(series)

    out = pd.DataFrame(rows).T.loc[[v for v, _ in ROW_PLAN]][STAT_COLS].round(4)
    out.index.name = None

    # The five rating buckets partition the sample — assert before publishing.
    buckets = ["ig_grade", "BB_grade", "B_grade", "CCC_below", "non_rated_suppl_all"]
    share = float(sum(out.loc[b, "Mean"] for b in buckets))
    assert abs(share - 1.0) < 1e-3, f"rating buckets sum to {share:.4f}, expected 1.0"
    print(f"\n  rating buckets sum to {share:.4f} ✅")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        out.to_excel(xw, sheet_name="Table1A", index=True, startrow=0)
        ws = xw.sheets["Table1A"]
        ws.write(len(out) + 2, 0, f"N = {len(smp):,} (Model 7 estimation sample)")
        ws.write(len(out) + 3, 0, "ig_grade is the omitted reference category in the regressions; "
                                  "reported here for completeness.")
        ws.set_column(0, 0, 26)
        ws.set_column(1, len(STAT_COLS), 12)

    print(f"\nSaved → {OUT_FILE}  ({len(out)} rows, N = {len(smp):,})")
    print(out.to_string())


if __name__ == "__main__":
    run()
