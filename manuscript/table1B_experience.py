"""
table1B_experience.py  —  Table 1, Panel B: lender-experience distributions
===========================================================================

Manuscript Table 1 Panel B. Reports P10, P25, P50, P75 and P90 for the 28 RQ2
lender-experience test variables, at BOTH the 12-month and the 36-month lookback
window (56 rows total), on the same Model 7 estimation sample as Panel A.

Structure follows the reference table (output/tables/Tables 08-13-26.xlsx, sheet
"Table1", from row 34 on): the 12-month block first, then the 36-month block; within
each, seven event families in the order AEC, FR, IC, GC, LF, MI, SP, each broken into
the four Top5 × Relatedness cells in the order NonTop5_Unrelated, Top5_Unrelated,
NonTop5_Related, Top5_Related.

Row names keep the CODE's own identifiers (AEC_Top5_Related, …) rather than the
reference's prose labels, matching Panel A's no-relabeling convention.

Notes on construction — identical to 07_rq2_experience / 09 Diagnostic 1:
  • Variables are reported in REGRESSION FORM, i.e. log(1+x). They are counts of
    irregularity events at the syndicate's lenders in their OTHER borrowers, so the
    distributions are heavily zero-inflated — hence percentiles rather than Panel A's
    mean/std, and hence no winsorization (the log already tames the tail).
  • IC is the element-wise MAX of the auditor and manager internal-control weakness
    counts (consolidated 2026-07-16), so it is built from two source columns.
  • ALL-LENDER sample (suffix "a"). The lead-arranger variant is a separate column
    block in 07's table and is not part of Panel B.
  • Top5 / NonTop5 splits the count by whether the lender that experienced the event is
    a top-5 lender; a contract with no qualifying lender contributes 0, while a genuinely
    missing count stays NaN and drops from that variable's percentiles.

Input : data/contracts.parquet
Output: output/tables/table1B.xlsx
"""

import numpy as np
import pandas as pd
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent   # code/ — this script lives in code/manuscript/
DATA_DIR = REPO_DIR.parent / "data"
OUT_DIR  = REPO_DIR.parent / "output" / "tables"
OUT_FILE = OUT_DIR / "table1B.xlsx"

# ── Sample spec — kept in lock-step with 06/07 (bucket ratings) ──────────────────
DETERMINANTS = ["accounting_policy", "offbslease", "BB_grade", "B_grade", "CCC_below",
                "non_rated_suppl_all", "relationship_freq"]
CONTROLS = [
    "maturity", "log_lender_count", "log_interest", "log_deal_amount", "perf_pricing",
    "fin_covenant_count", "gen_covenant_count", "secured",
    "size", "profitability", "bsfixed", "liabilities", "logage", "btm", "capex",
    "loss", "rand", "divyield",
    "log_bond_count", "bond_proceeds_scaled",
]
DV_SCORES = ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
             "claude_VAR_SCORE", "claude_RES_SCORE"]

# ── Experience-variable spec (order = the reference table's order) ───────────────
# (source stem(s), grouping root, row-label prefix).  A LIST of stems means the count
# is the element-wise row-wise MAX of those columns (IC = max(auditor, manager)).
EVENTS = [
    ("est_nc_sum_max_aec",                                   "aec",  "AEC"),
    ("res_sum_adv1_max_fr",                                  "fr",   "FR"),
    (["ic_sum_a_ineff_max_ic", "ic_sum_m_ineff_max_ic"],     "ic",   "IC"),
    ("aqrm_gc_sum_max_aqrm",                                 "aqrm", "GC"),
    ("aqrm_lf_sum_max_aqrm",                                 "aqrm", "LF"),
    ("aqrm_mi_sum_max_aqrm",                                 "aqrm", "MI"),
    ("sp_default_sum_max_sp",                                "sp",   "SP"),
]
DIMS = ["NonTop5_Unrelated", "Top5_Unrelated", "NonTop5_Related", "Top5_Related"]
WINDOWS = ["12", "36"]          # reference order: 12-month block, then 36-month
LENDER_SUFFIX = "a"             # all lenders

STAT_COLS = ["P10", "P25", "P50", "P75", "P90"]


def compute_stats(s: pd.Series) -> dict:
    s = s.dropna().astype(float)
    return {
        "P10": s.quantile(0.10),
        "P25": s.quantile(0.25),
        "P50": s.quantile(0.50),
        "P75": s.quantile(0.75),
        "P90": s.quantile(0.90),
    }


def build_experience_one(df: pd.DataFrame, stem, grp_root: str, window: str) -> pd.DataFrame:
    """The four log(1+x) test variables for one event family × window (all lenders)."""
    out = {}
    for rel_code, rel_name in [("u", "Unrelated"), ("r", "Related")]:
        suf = f"{rel_code}{LENDER_SUFFIX}"
        if isinstance(stem, (list, tuple)):
            src = [f"{s}_{suf}_{window}" for s in stem]
            # skipna=False → any NaN among the listed counts stays NaN (genuine missingness).
            val = df[src].max(axis=1, skipna=False).astype(float)
        else:
            val = df[f"{stem}_{suf}_{window}"].astype(float)
        grp = df[f"selected_grouping_{grp_root}_{suf}_{window}"]

        # .eq(True)/.eq(False) on the object dtype gives False for None → 0 in both buckets.
        top5    = np.where(grp.eq(True),  val, 0.0)
        nontop5 = np.where(grp.eq(False), val, 0.0)
        missing = val.isna().to_numpy()          # np.where above would have mapped NaN → 0
        top5[missing] = np.nan
        nontop5[missing] = np.nan

        out[f"NonTop5_{rel_name}"] = np.log1p(nontop5)
        out[f"Top5_{rel_name}"]    = np.log1p(top5)
    return pd.DataFrame(out, index=df.index)[DIMS]


def build_sample() -> pd.DataFrame:
    """07's Model 7 estimation sample — the same sample Panel A describes."""
    print("Loading contracts …")
    df = pd.read_parquet(DATA_DIR / "contracts.parquet")
    print(f"  {len(df):,} rows × {df.shape[1]} cols")

    df = df[df["claude_is_debt_contract"] == "Y"].copy()
    df = df[df[DV_SCORES].notna().all(axis=1)].copy()
    df["ALL_score_dummy"] = (df[DV_SCORES] > 0).any(axis=1).astype(float)

    gaap, freeze = df["claude_GAAP_OVERRIDE_SCORE"], df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan

    need   = ["ALL_score_dummy"] + DETERMINANTS + CONTROLS
    finite = np.isfinite(df[need].to_numpy(dtype=float)).all(axis=1)
    smp    = df[pd.Series(finite, index=df.index) & df["gvkey"].notna()].copy()
    print(f"  {len(smp):,} rows in the Model 7 estimation sample")
    return smp


def run() -> None:
    smp = build_sample()

    index, rows = [], []
    for window in WINDOWS:
        index.append(f"{window}-Months")                 # block header
        rows.append({c: np.nan for c in STAT_COLS})
        for stem, grp_root, label in EVENTS:
            exp = build_experience_one(smp, stem, grp_root, window)
            for dim in DIMS:
                index.append(f"{label}_{dim}")
                rows.append(compute_stats(exp[dim]))

    out = pd.DataFrame(rows, index=index)[STAT_COLS].round(4)
    out.index.name = None

    n_data = len(WINDOWS) * len(EVENTS) * len(DIMS)
    assert len(out) == n_data + len(WINDOWS), f"expected {n_data} data rows + block headers"
    print(f"\n  {n_data} data rows = {len(WINDOWS)} windows × {len(EVENTS)} events "
          f"× {len(DIMS)} dims ✅")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        out.to_excel(xw, sheet_name="Table1B", index=True)
        ws = xw.sheets["Table1B"]
        ws.write(len(out) + 2, 0, f"N = {len(smp):,} (Model 7 estimation sample)")
        ws.write(len(out) + 3, 0, "All variables in regression form, log(1 + event count); "
                                  "all-lender sample. IC = max(auditor, manager).")
        ws.set_column(0, 0, 30)
        ws.set_column(1, len(STAT_COLS), 12)

    print(f"\nSaved → {OUT_FILE}  ({len(out)} rows, N = {len(smp):,})")
    print(out.to_string())


if __name__ == "__main__":
    run()
