"""
table6_supplement.py  —  Table 6 supplement: correlations + baseline rates
===========================================================================

Supplementary statistics for Table 6, on the SAME estimation samples the Table 6 panels
use. Four sheets:

  Panel A correlations   ±3-year window (N = 1,704) — correlation matrix over the five
                         recognition dummies + accounting_policy and its two components
  Panel B correlations   ±5-year window (N = 2,602) — same matrix
  Panel A baseline       ±3-year window — score-level counts for each recognition component
                         in the UNTREATED cell: gaap_override = 0, freeze = 0,
                         post_adoption = 0
  Panel B baseline       ±5-year window — same
  Panel A policy 2x2     ±3-year window — two stacked 2×2 tables over accounting_policy
                         (0/1) × post_adoption (pre/post): contract counts, the cell sizes
                         Table 6 Panels A and C identify the post × accounting_policy
                         interaction off, and the mean of ALL_score_dummy in each cell (%),
                         i.e. the raw difference-in-differences the ALL column estimates
                         with FE and controls, with the accounting_policy 1 − 0 difference
                         and its two-proportion z-test at the foot of each column, plus a
                         DiD column differencing the post and pre gaps
  Panel B policy 2x2     ±5-year window — the same two tables (Table 6 Panels B and D)

The baseline sheets give the pre-adoption, no-accounting-policy-clause composition against
which Table 6's post × accounting_policy coefficients should be read: a −0.157 shift means
little without knowing the starting rate. Counts are reported by RAW SCORE LEVEL (0–3) for
the four components, so the intensive margin is visible and not collapsed into a dummy;
the final row gives the ALL dummy as a 0/1 split (levels 2–3 are not defined for it, shown
as an em dash). VAR-RES is max(claude_VAR_SCORE, claude_RES_SCORE), matching DV_SPECS. Rows
sum to the cell size.

SAMPLE. Reproduced exactly as in table6_rq3_asc842.py: scored debt contracts → adopting
firms only (adoption_date present) → symmetric ±window around adoption_date → all Model-7
regressors, the DV and gvkey non-missing. Fixed effects and lender data are NOT needed —
FE dummies are always finite, so they never gate a row — which is why this script does not
read dealscan_raw.parquet. N is asserted against Table 6's panels before writing.

CORRELATIONS. Spearman, upper triangle only, coefficient + stars (*** p<0.01, ** p<0.05,
* p<0.10), diagonal 1.00 — matching table1C. All eight variables are binary, so Spearman
and Pearson coincide (05's upper/lower split would print the same number twice); the script
verifies that identity before writing. Rows/columns keep the ORIGINAL variable names.

⚠ accounting_policy is the OR of gaap_override and freeze, so its correlations with them are
mechanical, not empirical — across the full scored sample only 1 contract of 14,584 has
freeze without gaap_override, which is why accounting_policy ≈ gaap_override.

Input : data/fulldata.parquet
Output: output/tables/table6_supplement.xlsx
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr

REPO_DIR  = Path(__file__).resolve().parent.parent   # code/ — this script lives in code/manuscript/
DATA_DIR  = REPO_DIR.parent / "data"
OUT_DIR   = REPO_DIR.parent / "output" / "tables"
OUT_FILE  = OUT_DIR / "table6_supplement.xlsx"
CONTRACTS = DATA_DIR / "fulldata.parquet"

# (panel, ± years, expected N — asserted against Table 6)
WINDOWS = [("Panel A", 3, 1704), ("Panel B", 5, 2602)]

DV_SCORES = ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
             "claude_VAR_SCORE", "claude_RES_SCORE"]
DV_SPECS = [
    ("SLB",     ["claude_SLB_SCORE"]),
    ("SYN",     ["claude_SYN_SCORE"]),
    ("OPL",     ["claude_OPL_SCORE"]),
    ("VAR-RES", ["claude_VAR_SCORE", "claude_RES_SCORE"]),
    ("ALL",     DV_SCORES),
]
DID_COL = "DiD (Post - Pre)"
RECOGNITION = ["ALL_score_dummy", "SLB_score_dummy", "SYN_score_dummy",
               "OPL_score_dummy", "VAR-RES_score_dummy"]
CORR_VARS   = RECOGNITION + ["accounting_policy", "gaap_override", "freeze"]

# Table 6's Model-7 regressor set — used here only to reproduce its estimation sample.
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
POST = "post_adoption"


def _stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def _corr_p(func, x, y):
    res = func(x, y)
    try:
        return float(res.statistic), float(res.pvalue)
    except AttributeError:
        return float(res[0]), float(res[1])


def _fmt(r: float, p: float) -> str:
    s = f"{r:.2f}"
    return ("0.00" if s == "-0.00" else s) + _stars(p)


def build_corr_matrix(data: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Spearman upper triangle, diagonal 1.00, lower triangle blank (as in table1C)."""
    arrs = {v: data[v].to_numpy(dtype=float) for v in variables}
    M = pd.DataFrame("", index=variables, columns=variables, dtype=object)
    for i, vi in enumerate(variables):
        for j, vj in enumerate(variables):
            if i == j:
                M.iat[i, j] = "1.00"
            elif i < j:
                M.iat[i, j] = _fmt(*_corr_p(spearmanr, arrs[vi], arrs[vj]))
    M.index.name = "Spearman"
    return M


def build_sample(window_years: int) -> pd.DataFrame:
    """Table 6's estimation sample for one window (see module docstring)."""
    df = pd.read_parquet(CONTRACTS)
    df = df[df["claude_is_debt_contract"] == "Y"]
    df = df[df[DV_SCORES].notna().all(axis=1)].copy()

    for sheet, cols in DV_SPECS:
        df[f"{sheet}_score_dummy"] = (df[cols] > 0).any(axis=1).astype(float)

    gaap, freeze = df["claude_GAAP_OVERRIDE_SCORE"], df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan
    df["gaap_override"] = (gaap > 0).astype(float)
    df.loc[gaap.isna(), "gaap_override"] = np.nan
    df["freeze"] = (freeze > 0).astype(float)
    df.loc[freeze.isna(), "freeze"] = np.nan

    adopt = pd.to_datetime(df["adoption_date"], errors="coerce")
    df[POST] = np.where(adopt.isna(), np.nan,
                        (df["tranche_active_date"] >= adopt).astype(float))
    df = df[df[POST].notna()].copy()             # adopting firms only

    adopt = pd.to_datetime(df["adoption_date"], errors="coerce")
    df = df[((df["tranche_active_date"] >= adopt - pd.DateOffset(years=window_years))
             & (df["tranche_active_date"] <= adopt + pd.DateOffset(years=window_years)))].copy()

    # Regressor completeness — the same gate stabilize_design applies in table6.
    finite = np.isfinite(df[[POST] + DETERMINANTS + CONTROLS].to_numpy(dtype=float)).all(axis=1)
    keep   = (pd.Series(finite, index=df.index)
              & df[RECOGNITION].notna().all(axis=1) & df["gvkey"].notna())
    return df[keep].copy()


LEVELS   = [0, 1, 2, 3]
LEVEL_COLS = [f"level_{i}" for i in LEVELS]


def build_baseline(smp: pd.DataFrame):
    """Score-level counts in the untreated cell: no clause, pre-adoption.

    One row per recognition component, counts by RAW score level 0–3 (VAR-RES = the
    element-wise max of its two source scores, as in DV_SPECS), then the ALL dummy as a
    0/1 split — levels 2–3 do not exist for a dummy and are shown as an em dash.
    """
    cell = smp[(smp["gaap_override"] == 0) & (smp["freeze"] == 0) & (smp[POST] == 0)]
    rows = {}
    for name, cols in DV_SPECS:
        if name == "ALL":
            continue
        lvl = cell[cols].max(axis=1) if len(cols) > 1 else cell[cols[0]]
        vc  = lvl.value_counts().reindex(LEVELS, fill_value=0)
        rows[name] = {c: int(vc[i]) for c, i in zip(LEVEL_COLS, LEVELS)}
    d = cell["ALL_score_dummy"].value_counts().reindex([0, 1], fill_value=0)
    rows["ALL (0/1)"] = {"level_0": int(d[0]), "level_1": int(d[1]),
                         "level_2": "—", "level_3": "—"}

    out = pd.DataFrame(rows).T[LEVEL_COLS]
    out.index.name = "component"
    # Every component row must account for the whole cell.
    for r in out.index[:-1]:
        assert sum(int(out.loc[r, c]) for c in LEVEL_COLS) == len(cell), (
            f"{r} levels sum to != cell size {len(cell)}")
    return out, len(cell)


def build_policy_2x2(smp: pd.DataFrame) -> pd.DataFrame:
    """Contract counts in the accounting_policy × post_adoption 2×2, with margins.

    These four cells are what Table 6's post × accounting_policy interaction is identified
    off: the treated-post cell is the binding one, so its size bounds how much the headline
    coefficient can be trusted.
    """
    tab = (pd.crosstab(smp["accounting_policy"], smp[POST])
             .reindex(index=[0.0, 1.0], columns=[0.0, 1.0], fill_value=0))
    tab.index   = ["accounting_policy = 0", "accounting_policy = 1"]
    tab.columns = ["Pre (post_adoption = 0)", "Post (post_adoption = 1)"]
    tab["Total"] = tab.sum(axis=1)
    tab.loc["Total"] = tab.sum(axis=0)
    tab = tab.astype(int)
    tab.index.name = "contracts"
    assert int(tab.loc["Total", "Total"]) == len(smp), (
        f"2x2 totals {int(tab.loc['Total', 'Total']):,} != sample {len(smp):,}")
    return tab


def _two_prop_z(x1: int, n1: int, x0: int, n0: int):
    """Pooled two-proportion z-test: H0 p1 == p0, two-sided. Returns (diff in pp, z, p)."""
    p1, p0 = x1 / n1, x0 / n0
    pool = (x1 + x0) / (n1 + n0)
    se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n0))
    z = (p1 - p0) / se if se > 0 else np.nan
    return 100 * (p1 - p0), z, 2 * norm.sf(abs(z))


def _did_z(cells: dict):
    """z on (post gap − pre gap) for four independent proportions.

    Unpooled: under H0 the two gaps are equal but no common proportion is implied, so each
    cell contributes its own p(1−p)/n. Equivalent to the HC-robust z on the interaction of
    an OLS of the dummy on policy × post — and, like that regression, it ignores repeat
    borrowers (Table 6 clusters on gvkey).
    """
    d_pre  = cells[(1, 0)][0] - cells[(0, 0)][0]
    d_post = cells[(1, 1)][0] - cells[(0, 1)][0]
    var = sum(pr * (1 - pr) / n for pr, n in cells.values())
    z = (d_post - d_pre) / np.sqrt(var) if var > 0 else np.nan
    return 100 * (d_post - d_pre), z, 2 * norm.sf(abs(z))


def build_policy_2x2_means(smp: pd.DataFrame, var: str = "ALL_score_dummy") -> pd.DataFrame:
    """Mean of `var` (in %) in each accounting_policy × post_adoption cell, with margins.

    ALL_score_dummy is the overall recognition dummy — the DV of Table 6's ALL column — so
    the four cells are the raw 2×2 difference-in-differences that column estimates with FE
    and controls. Margins are means over the pooled rows, not means of the cell means.
    """
    rows = {0.0: "accounting_policy = 0", 1.0: "accounting_policy = 1"}
    cols = {0.0: "Pre (post_adoption = 0)", 1.0: "Post (post_adoption = 1)"}
    out = pd.DataFrame(index=list(rows.values()) + ["All"],
                       columns=list(cols.values()) + ["All"], dtype=float)
    for rv, rl in rows.items():
        rmask = smp["accounting_policy"] == rv
        for cv, cl in cols.items():
            out.loc[rl, cl] = 100 * smp.loc[rmask & (smp[POST] == cv), var].mean()
        out.loc[rl, "All"] = 100 * smp.loc[rmask, var].mean()
    for cv, cl in cols.items():
        out.loc["All", cl] = 100 * smp.loc[smp[POST] == cv, var].mean()
    out.loc["All", "All"] = 100 * smp[var].mean()
    out = out.round(1).astype(object)

    # accounting_policy 1 − 0 within each period, then pooled — two-proportion z-test.
    masks = [(cl, smp[POST] == cv) for cv, cl in cols.items()]
    masks.append(("All", pd.Series(True, index=smp.index)))
    diffs, zs = {}, {}
    for cl, cmask in masks:
        g1 = smp.loc[cmask & (smp["accounting_policy"] == 1), var]
        g0 = smp.loc[cmask & (smp["accounting_policy"] == 0), var]
        d, z, pv = _two_prop_z(int(g1.sum()), len(g1), int(g0.sum()), len(g0))
        diffs[cl], zs[cl] = f"{d:.1f}{_stars(pv)}", f"{z:.2f}"
    # (post gap − pre gap): the raw DiD Table 6's post × accounting_policy estimates.
    cells = {(int(rv), int(cv)): (smp.loc[(smp["accounting_policy"] == rv) & (smp[POST] == cv), var].mean(),
                                  int(((smp["accounting_policy"] == rv) & (smp[POST] == cv)).sum()))
             for rv in rows for cv in cols}
    did, z_did, p_did = _did_z(cells)
    diffs[DID_COL], zs[DID_COL] = f"{did:.1f}{_stars(p_did)}", f"{z_did:.2f}"

    out[DID_COL] = ""
    out.loc["Difference (1 - 0), pp"] = pd.Series(diffs)
    out.loc["z"] = pd.Series(zs)
    out.index.name = f"mean of {var} (%)"
    return out


def run() -> None:
    sheets = {}
    for panel, yrs, expect_n in WINDOWS:
        smp = build_sample(yrs)
        print(f"\n{panel}  ±{yrs}y: N = {len(smp):,} (Table 6 reports {expect_n:,})")
        assert len(smp) == expect_n, (
            f"{panel}: sample is {len(smp):,}, expected {expect_n:,} — the supplement must "
            f"describe exactly the rows Table 6 estimates on")

        # Binary variables ⇒ Spearman == Pearson. Verify rather than assume.
        a = {v: smp[v].to_numpy(float) for v in CORR_VARS}
        worst = max(abs(_corr_p(spearmanr, a[x], a[y])[0] - _corr_p(pearsonr, a[x], a[y])[0])
                    for i, x in enumerate(CORR_VARS) for y in CORR_VARS[i + 1:])
        print(f"  max |Spearman − Pearson| over the 28 pairs = {worst:.2e} ✅")
        sheets[f"{panel} correlations"] = build_corr_matrix(smp, CORR_VARS)

        base, n_cell = build_baseline(smp)
        print(f"  baseline cell (gaap_override=0, freeze=0, post=0): N = {n_cell:,}")
        print("    " + f"{'component':<12}" + "".join(f"{c:>9}" for c in LEVEL_COLS))
        for r in base.index:
            print("    " + f"{r:<12}" + "".join(f"{base.loc[r, c]:>9}" for c in LEVEL_COLS))
        sheets[f"{panel} baseline"] = base
        sheets[f"{panel} baseline"].attrs["n_cell"] = n_cell

        two  = build_policy_2x2(smp)
        mean = build_policy_2x2_means(smp)
        for title, tab, fmt in [("Contracts", two, ",d"),
                                ("Mean of ALL_score_dummy (%)", mean, "")]:
            print(f"  {title} — accounting_policy x post_adoption:")
            print("    " + f"{'':<24}" + "".join(f"{c:>26}" for c in tab.columns))
            for r in tab.index:
                print("    " + f"{r:<24}"
                      + "".join(f"{tab.loc[r, c]:>26{fmt}}" for c in tab.columns))
        sheets[f"{panel} policy 2x2"] = [("Contracts", two),
                                         ("Mean of ALL_score_dummy (%)", mean)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    order = ["Panel A correlations", "Panel B correlations",
             "Panel A baseline", "Panel B baseline",
             "Panel A policy 2x2", "Panel B policy 2x2"]
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        for name in order:
            # A sheet is either one table or a list of (title, table) blocks stacked down it.
            blocks = sheets[name]
            blocks = blocks if isinstance(blocks, list) else [(None, blocks)]
            titles, row = [], 0
            for title, tab in blocks:
                if title:
                    titles.append((row, title))
                    row += 1
                tab.to_excel(xw, sheet_name=name, startrow=row)
                row += len(tab) + 2          # header + body + one blank line
            ws = xw.sheets[name]
            for r, title in titles:
                ws.write(r, 0, title)
            ws.set_column(0, 0, 26)
            ws.set_column(1, max(t.shape[1] for _, t in blocks), 26)
            panel, yrs, expect_n = next(w for w in WINDOWS if w[0] == name.split()[0] + " " + name.split()[1])
            if "correlations" in name:
                ws.write(row, 0, f"N = {expect_n:,}  (±{yrs}-year window, Table 6 {panel})")
                ws.write(row + 1, 0, "Spearman; all variables binary so Pearson is identical. "
                                     "*** p<0.01, ** p<0.05, * p<0.10.")
                ws.write(row + 2, 0, "accounting_policy is the OR of gaap_override and freeze — "
                                     "those correlations are mechanical.")
            elif "policy 2x2" in name:
                twin = {"Panel A": "A and C", "Panel B": "B and D"}[panel]
                ws.write(row, 0, f"±{yrs}-year window (the sample of Table 6 Panels "
                                 f"{twin}, N = {expect_n:,}).")
                ws.write(row + 1, 0, "Rows: accounting_policy (the OR of gaap_override and "
                                     "freeze). Columns: post_adoption, relative to the firm's "
                                     "ASC 842 adoption date.")
                ws.write(row + 2, 0, "ALL_score_dummy = any of the five recognition scores > 0, "
                                     "the DV of Table 6's ALL column. Margins are means over the "
                                     "pooled rows, not means of the cell means.")
                ws.write(row + 3, 0, "Difference = accounting_policy 1 - 0 within the column, in "
                                     "percentage points; z from a pooled two-proportion z-test "
                                     "(two-sided). *** p<0.01, ** p<0.05, * p<0.10. Contracts are "
                                     "treated as independent - no gvkey clustering, unlike Table 6.")
                ws.write(row + 4, 0, f"{DID_COL} = post gap minus pre gap, the raw "
                                     "difference-in-differences; z unpooled (each of the four "
                                     "cells contributes p(1-p)/n), matching the HC-robust z on "
                                     "the interaction in an OLS of the dummy on policy x post.")
            else:
                ws.write(row, 0, f"Untreated cell of the ±{yrs}-year window "
                                 f"(Table 6 {panel}, N = {expect_n:,}): "
                                 f"gaap_override = 0, freeze = 0, post_adoption = 0.")
                ws.write(row + 1, 0, f"Cell size N = {tab.attrs.get('n_cell', ''):,}. "
                                     "Counts by raw score level; VAR-RES = max(VAR, RES). "
                                     "ALL is a 0/1 dummy, so levels 2-3 do not apply.")
    print(f"\nSaved → {OUT_FILE}  (sheets: {', '.join(order)})")


if __name__ == "__main__":
    run()
