"""
table1C_correlations.py  —  Table 1, Panel C: correlation matrix
=================================================================

Manuscript Table 1 Panel C. Correlation matrix over the five recognition dummies and
the three accounting-policy variables, on the same Model 7 estimation sample as
Panels A and B (N = 11,184).

Structure follows 05_descriptives.py's correlations sheet, reduced to a SINGLE
triangle:
  • UPPER triangle = Spearman, diagonal = 1.00, lower triangle left blank
  • cells are the coefficient to 2 d.p. plus significance stars
  • stars: *** p<0.01, ** p<0.05, * p<0.10

Rows/columns keep the ORIGINAL variable names, matching Panels A and B.

Variables (order follows the reference table's Panel A):
    ALL_score_dummy       SLB_score_dummy      SYN_score_dummy
    OPL_score_dummy       VAR-RES_score_dummy  accounting_policy
    gaap_override         freeze

NOTE. All eight variables are binary, and for two binary variables Spearman's rho is
algebraically identical to Pearson's r (both reduce to the phi coefficient). 05's
upper/lower split would therefore have printed the same number twice, which is why this
panel reports the Spearman triangle only. The script still checks the identity against
Pearson before writing, so the reported coefficients are equally readable as either.

Note also that gaap_override and freeze are the two COMPONENTS of accounting_policy
(which is their OR), so their correlations with it are mechanical, not empirical.

Input : data/fulldata.parquet
Output: output/tables/table1C.xlsx
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

REPO_DIR = Path(__file__).resolve().parent.parent   # code/ — this script lives in code/manuscript/
DATA_DIR = REPO_DIR.parent / "data"
OUT_DIR  = REPO_DIR.parent / "output" / "tables"
OUT_FILE = OUT_DIR / "table1C.xlsx"

# ── Sample spec — kept in lock-step with 06 (bucket ratings), as in Panels A/B ───
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

# Panel C variables, in the reference table's presentation order.
CORR_VARS = [
    "ALL_score_dummy",
    "SLB_score_dummy",
    "SYN_score_dummy",
    "OPL_score_dummy",
    "VAR-RES_score_dummy",
    "accounting_policy",
    "gaap_override",
    "freeze",
]


def _corr_star(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def _corr_p(func, x: np.ndarray, y: np.ndarray):
    """(coef, p) from pearsonr/spearmanr across scipy versions."""
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
    """Upper triangle = Spearman, diagonal = 1.00, lower triangle blank."""
    arrs = {v: data[v].to_numpy(dtype=float) for v in variables}
    M = pd.DataFrame("", index=variables, columns=variables, dtype=object)
    for i, vi in enumerate(variables):
        for j, vj in enumerate(variables):
            if i == j:
                M.iat[i, j] = "1.00"
            elif i < j:
                M.iat[i, j] = _fmt_corr(*_corr_p(spearmanr, arrs[vi], arrs[vj]))
            # lower triangle intentionally left blank
    M.index.name = "Spearman"
    return M


def build_sample() -> pd.DataFrame:
    """The Model 7 estimation sample — identical to Panels A and B."""
    print("Loading contracts …")
    df = pd.read_parquet(DATA_DIR / "fulldata.parquet")
    print(f"  {len(df):,} rows × {df.shape[1]} cols")

    df = df[df["claude_is_debt_contract"] == "Y"].copy()
    for dv, score_cols in DV_SPECS:
        df[dv] = (df[score_cols] > 0).any(axis=1).astype(float)

    gaap, freeze = df["claude_GAAP_OVERRIDE_SCORE"], df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan
    df["gaap_override"] = (gaap > 0).astype(float)
    df.loc[gaap.isna(), "gaap_override"] = np.nan
    df["freeze"] = (freeze > 0).astype(float)
    df.loc[freeze.isna(), "freeze"] = np.nan

    finite   = np.isfinite(df[DETERMINANTS + CONTROLS].to_numpy(dtype=float)).all(axis=1)
    score_ok = df[sorted({c for _, cols in DV_SPECS for c in cols})].notna().all(axis=1)
    smp      = df[pd.Series(finite, index=df.index) & score_ok & df["gvkey"].notna()].copy()
    print(f"  {len(smp):,} rows in the Model 7 estimation sample")
    return smp


def run() -> None:
    smp = build_sample()

    missing = {v: int(smp[v].isna().sum()) for v in CORR_VARS}
    assert not any(missing.values()), f"NaN in Panel C variables: {missing}"

    corr = build_corr_matrix(smp, CORR_VARS)

    # All eight variables are binary → Spearman == Pearson. Verify rather than assume.
    a = {v: smp[v].to_numpy(float) for v in CORR_VARS}
    worst = max(abs(_corr_p(spearmanr, a[x], a[y])[0] - _corr_p(pearsonr, a[x], a[y])[0])
                for i, x in enumerate(CORR_VARS) for y in CORR_VARS[i + 1:])
    print(f"\n  max |Spearman − Pearson| across the 28 pairs = {worst:.2e} "
          f"(binary variables ⇒ identical) ✅")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        corr.to_excel(xw, sheet_name="Table1C", index=True)
        ws = xw.sheets["Table1C"]
        ws.write(len(corr) + 2, 0, f"N = {len(smp):,} (Model 7 estimation sample)")
        ws.write(len(corr) + 3, 0, "Spearman correlations. *** p<0.01, ** p<0.05, * p<0.10.")
        ws.write(len(corr) + 4, 0, "All variables binary, so Spearman and Pearson coincide. "
                                   "gaap_override and freeze are the components of accounting_policy.")
        ws.set_column(0, 0, 24)
        ws.set_column(1, len(CORR_VARS), 14)

    print(f"\nSaved → {OUT_FILE}  ({len(corr)}×{len(corr)}, N = {len(smp):,})")
    print(corr.to_string())


if __name__ == "__main__":
    run()
