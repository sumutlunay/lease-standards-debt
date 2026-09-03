"""
table5_rq2_controls.py  —  Table 5: RQ2 lender experience, with controls
=========================================================================

RQ2 lender-experience results in the WITH-CONTROLS specification — the full Model-7 spec —
in the same two-panel layout as Table 4: one sheet, two stacked panels, seven event-family
columns each.

  Window          : 12-month lookback only
  RHS             : the four experience test variables + the determinants + the 20 deal/firm
                    controls + the Model-7 fixed effects (Industry×Year + Borrower + Lender
                    multi-hot). Only the four TEST-VARIABLE rows are tabulated; the
                    determinant/control coefficients are estimated but NOT shown
  Panels          : A = all syndicate lenders' experience  (source suffix _{u,r}a)
                    B = lead arrangers' experience         (source suffix _{u,r}l)
  Standard errors : clustered by gvkey (needed for the stars, though t-stats are not printed)
  Cells           : coefficient + significance stars only — NO t-statistics
  R² / Adj. R²    : CENTERED, matching table2 / table3

SAMPLE is the Model-7 sample: a row enters iff the experience variables AND the determinants
AND the controls are all non-missing — the SAME rows as table4, so Tables 4 and 5 differ
ONLY in whether the controls sit on the right-hand side.

★ RATING SPEC. Credit quality enters as the BUCKETS (BB_grade / B_grade / CCC_below /
non_rated_suppl_all, with ig_grade the omitted reference) in place of the linear
num_rating_suppl_all, matching the rest of the pipeline (7985df5). ⚠ Unlike table4 — where
the rating only gated the sample — the determinants ARE regressors here, so this IS a
specification change: the design gains two columns and the tabulated test-variable
coefficients move.

Level variables are winsorized 1% both tails on the estimation sample (WINSOR_LEVEL_VARS),
exactly as in 07. Test variables are log(1+x) counts of irregularity events at the
syndicate's lenders in their OTHER borrowers, so they are not winsorized. Event families and
the test-variable construction are identical to exploratory/07_rq2_experience.py.

Rehash of exploratory/07c_rq2_table5.py, trimmed (console and docstring condensed).

★ LEAD-ARRANGER FE VARIANT (2026-09-02). Identical to code/manuscript/table5_rq2_controls.py in every
respect except ONE: the multi-hot lender fixed effects are built from the deal's credited
LEAD ARRANGERS instead of from all syndicate members. Same sample construction, same
regressors, same winsorization, same Industry×Year and Borrower blocks, same standard errors,
same row and column layout. A coefficient that moves between table5_leadfe.xlsx and table5.xlsx moved because of
the FE definition and nothing else.

DEFINITION — DealScan's `lead_arranger` field, per Tseng 11-10-25 p.25: "the second pool
assumes only lead arrangers identified by DealScan's 'Lead_Arranger' can do so", repeated in
the Table notes on pp.42-43. Credited names are resolved to `lender_parent_id`, so branches
and acquired institutions roll up to one bank — matching the parent-level identity of the
lender-experience files (`selected_parent_lender_id_*`) and of the all-lender FE this variant
replaces. See load_lender_lists() for the three-layer name resolution.

⚠ SUPERSEDES an earlier version of this script that defined lead lenders by DealScan
`primary_role` (an "Admin agent / Bookrunner / ..." role set). That definition had no basis in
the paper or in any upstream code and its results are void; this file replaces it in place.

⚠ The FE block is sparser than the all-lender baseline (~16 syndicate members per contract vs
a median of 1 credited arranger parent). Expect fewer surviving FE columns after the singleton
guard and weaker absorption — a lower R² here is mechanical, not a finding.

⚠ Contracts with no credited arranger keep an all-zero `l_` block rather than dropping, exactly
as the baseline treats contracts with no DealScan lender match. N is therefore unchanged.

Lives in code/exploratory/ — a robustness check, not part of the manuscript set.

Input : data/fulldata.parquet, data/dealscan_raw.parquet
Output: output/tables/table5_leadfe.xlsx  (single sheet 'Table 5', two panels)
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_DIR = Path(__file__).resolve().parent.parent   # code/ — this script lives in code/exploratory/
DATA_DIR = REPO_DIR.parent / "data"
OUT_DIR  = REPO_DIR.parent / "output" / "tables"
OUT_FILE = OUT_DIR / "table5_leadfe.xlsx"
SHEET    = "Table 5"

MERGE_KEYS = ["borrower_id", "tranche_active_date"]
DV_SHEET   = "ALL"
DV_SCORES  = ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
              "claude_VAR_SCORE", "claude_RES_SCORE"]
WINDOW     = "12"

# (prefix, count-stem(s), grouping root, label). A LIST of stems → element-wise row-wise MAX.
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
LENDER_SAMPLES = [
    ("Panel A: All lenders' experience (12-month window)",  "a"),
    ("Panel B: Lead lenders' experience (12-month window)", "l"),
]
EXPERIENCE_BASE = ["NonTop5_Unrelated", "Top5_Unrelated", "NonTop5_Related", "Top5_Related"]

# ★ These ARE regressors here (estimated, not tabulated). Bucket ratings; ig_grade omitted.
DETERMINANTS = ["accounting_policy", "offbslease", "BB_grade", "B_grade", "CCC_below",
                "non_rated_suppl_all", "relationship_freq"]
CONTROLS = [
    "maturity", "log_lender_count", "log_interest", "log_deal_amount", "perf_pricing",
    "fin_covenant_count", "gen_covenant_count", "secured",
    "size", "profitability", "bsfixed", "liabilities", "logage", "btm", "capex",
    "loss", "rand", "divyield",
    "log_bond_count", "bond_proceeds_scaled",
]
# Non-logged, non-dummy level variables winsorized 1% both tails on the estimation sample.
WINSOR_LEVEL_VARS = [
    "offbslease",
    "fin_covenant_count", "gen_covenant_count",
    "profitability", "bsfixed", "liabilities", "btm", "capex", "rand", "divyield",
    "bond_proceeds_scaled",
]


def exp_names(prefix: str) -> list:
    return [f"{prefix}_{v}" for v in EXPERIENCE_BASE]


def event_code(event: tuple) -> str:
    """'AEC: …' → 'AEC'."""
    return event[3].split(":")[0].strip()


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


# ── Lead arrangers → lender_parent_id ────────────────────────────────────────────
# DealScan credits lead arrangers in the deal-level `lead_arranger` string (Tseng
# 11-10-25, p.25: "the second pool assumes only lead arrangers identified by DealScan's
# 'Lead_Arranger' can do so"). Names are resolved to `lender_parent_id` so branches and
# acquired institutions roll up to one bank, matching the parent-level identity of the
# `full_parent_event_selected_*` files (`selected_parent_lender_id_*`).

_SHARE_RE = re.compile(r"\s*\d{1,3}(?:\.\d+)?\s*%")


def parse_arrangers(s) -> list:
    """Credited arranger names: strip ' NN.NN%' share tokens, split on comma, dedupe."""
    if pd.isna(s) or str(s).strip() in ("", "None"):
        return []
    seen, out = set(), []
    for tok in _SHARE_RE.sub("", str(s)).split(","):
        t = tok.strip()
        if t and t != "None" and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_lender_lists() -> pd.Series:
    """Per deal-key: the de-duplicated `lender_parent_id`s of the credited lead arrangers.

    Names resolve in three layers, most faithful first:
      1. the deal's OWN lender rows, matched on `lender_name`      (native id, no heuristic)
      2. the deal's own lender rows, matched on `lender_parent_name`
      3. a global modal name→parent crosswalk built over all of dealscan_raw   (fallback)
    Layers 1-2 and layer 3 agree on 99.96% of the name-instances where both resolve, so the
    fallback only fills gaps rather than steering the result.
    """
    ds = pd.read_parquet(DATA_DIR / "dealscan_raw.parquet",
                         columns=MERGE_KEYS + ["lender_parent_id", "lender_parent_name",
                                               "lender_name", "lead_arranger"])
    ds["tranche_active_date"] = pd.to_datetime(ds["tranche_active_date"], errors="coerce")
    ds = ds.dropna(subset=MERGE_KEYS)

    lend = ds.dropna(subset=["lender_parent_id"]).copy()
    lend["lender_parent_id"] = lend["lender_parent_id"].astype(int)

    wd_b = (lend.groupby(MERGE_KEYS + ["lender_name"])["lender_parent_id"].first()
            .rename("pid_b").reset_index().rename(columns={"lender_name": "name"}))
    wd_p = (lend.groupby(MERGE_KEYS + ["lender_parent_name"])["lender_parent_id"].first()
            .rename("pid_p").reset_index().rename(columns={"lender_parent_name": "name"}))
    modal = lambda s_: s_.mode().iat[0]
    xwalk = {**lend.groupby("lender_parent_name")["lender_parent_id"].agg(modal).to_dict(),
             **lend.groupby("lender_name")["lender_parent_id"].agg(modal).to_dict()}

    ds["lead_arranger"] = ds["lead_arranger"].where(
        ds["lead_arranger"].astype(str).str.strip().ne("None"))
    # UNION, not .first(). `lead_arranger` is a deal-level string, but our contract key
    # (borrower_id x tranche_active_date) is coarser than DealScan's tranche: 8.74% of keys
    # carry >1 distinct value because several tranches share a borrower-date, each with its
    # own credited arrangers. .first() took whichever row parquet order happened to put
    # first and silently dropped the rest — a one-directional error (on the keys where the
    # two differ, the .first() set was a strict SUBSET of the union 100% of the time,
    # missing 1.81 parent banks on average). Taking the union over the key's distinct
    # strings needs no tie-breaking and cannot invent an arranger.
    arr = (ds.groupby(MERGE_KEYS)["lead_arranger"]
             .apply(lambda s: sorted({v for v in s.dropna().unique()}))
             .apply(lambda L: [n for v in L for n in parse_arrangers(v)]))

    long = arr.explode().dropna().rename("name").reset_index()
    long = long.merge(wd_b, on=MERGE_KEYS + ["name"], how="left")
    long = long.merge(wd_p, on=MERGE_KEYS + ["name"], how="left")
    long["pid"] = long["pid_b"].fillna(long["pid_p"]).fillna(long["name"].map(xwalk))

    # within-deal = resolved by layer 1 OR layer 2 (the two overlap — do not add them)
    n_all = len(long)
    n_wd  = int((long["pid_b"].notna() | long["pid_p"].notna()).sum())
    n_bad = int(long["pid"].isna().sum())
    long = long.dropna(subset=["pid"])
    long["pid"] = long["pid"].astype(int)

    out = long.groupby(MERGE_KEYS)["pid"].apply(lambda s: sorted(set(s))).rename("lender_ids")
    print(f"  lead arrangers: {n_all:,} credited name-instances | within-deal {n_wd / n_all * 100:.1f}%"
          f" | unresolved {n_bad:,} | {len(out):,} deal-keys, "
          f"median {int(out.apply(len).median())} parent(s) per key")
    return out


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
    """CENTERED R²/Adj. R² for the no-intercept FE model; coefficients/SEs unaffected."""
    n  = int(res.nobs)
    r2 = 1.0 - res.ssr / res.centered_tss
    return r2, (1.0 - (1.0 - r2) * (n - 1) / (n - n_params) if n > n_params else float("nan"))


def winsorize_cols(frame: pd.DataFrame, cols: list, mask: pd.Series, p: float = 0.01):
    """Winsorize each column at [p, 1-p], bounds computed on the estimation sample."""
    out = frame.copy()
    for c in cols:
        if c in out.columns:
            lo, hi = out.loc[mask, c].quantile(p), out.loc[mask, c].quantile(1 - p)
            out[c] = out[c].clip(lower=lo, upper=hi)
    return out


def _stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


# ── Test-variable construction ───────────────────────────────────────────────────

def build_experience(df, prefix, stem, grp_root, sample_suf, window) -> pd.DataFrame:
    out = {}
    for rel_code, rel_name in [("u", "Unrelated"), ("r", "Related")]:
        suf = f"{rel_code}{sample_suf}"
        if isinstance(stem, (list, tuple)):
            # skipna=False → any NaN among the listed counts stays NaN (genuine missingness).
            val = df[[f"{s}_{suf}_{window}" for s in stem]].max(axis=1, skipna=False).astype(float)
        else:
            val = df[f"{stem}_{suf}_{window}"].astype(float)
        grp = df[f"selected_grouping_{grp_root}_{suf}_{window}"]

        top5    = np.where(grp.eq(True),  val, 0.0)
        nontop5 = np.where(grp.eq(False), val, 0.0)
        missing = val.isna().to_numpy()          # np.where above would map NaN → 0
        top5[missing] = np.nan
        nontop5[missing] = np.nan

        out[f"{prefix}_NonTop5_{rel_name}"] = np.log1p(nontop5)
        out[f"{prefix}_Top5_{rel_name}"]    = np.log1p(top5)
    return pd.DataFrame(out, index=df.index)[exp_names(prefix)]


# ── Base sample + FE matrices ────────────────────────────────────────────────────

def prepare_sample(df_full: pd.DataFrame, lender_lists: pd.Series):
    dv = f"{DV_SHEET}_score_dummy"
    df = df_full[df_full["claude_is_debt_contract"] == "Y"]
    df = df[df[DV_SCORES].notna().all(axis=1)].copy()

    df[dv]              = (df[DV_SCORES] > 0).any(axis=1).astype(float)
    df["contract_year"] = df["tranche_active_date"].dt.year
    gaap, freeze = df["claude_GAAP_OVERRIDE_SCORE"], df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan

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

    fe_labels = [f"Industry×Year FEs (SIC {sic_digits}-digit)", "Borrower FEs", "Lead-arranger FEs"]
    print(f"  {len(df):,} scored debt contracts | FEs — IY={X_fe.shape[1]} "
          f"Bor={fe_bor.shape[1]} Len={fe_lender.shape[1]}")
    return df, dv, X_fe, fe_bor, fe_lender, fe_labels


# ── One no-controls regression (one event family × one lender sample) ────────────

def run_experience_model(df, dv, X_fe, fe_bor, fe_lender, event, sample_suf) -> list:
    prefix, stem, grp_root, event_label = event
    experience = exp_names(prefix)
    regressors = experience + DETERMINANTS + CONTROLS      # FULL Model-7 spec

    df = df.copy()
    df[experience] = build_experience(df, prefix, stem, grp_root, sample_suf, WINDOW)

    # Model-7 estimation sample: experience + determinants + controls all non-missing.
    finite  = np.isfinite(df[regressors].to_numpy(dtype=float)).all(axis=1)
    m7_mask = pd.Series(finite, index=df.index) & df[dv].notna() & df["gvkey"].notna()
    sub     = df.index[m7_mask]

    # Winsorize levels on the estimation sample (test vars are log(1+x) → not winsorized).
    dfw = winsorize_cols(df, WINSOR_LEVEL_VARS, m7_mask)
    X_full = pd.concat([dfw.loc[sub, regressors].astype(float),
                        X_fe.loc[sub], fe_bor.loc[sub], fe_lender.loc[sub]], axis=1)
    X, y, cl = stabilize_design(X_full, dfw.loc[sub, dv], dfw.loc[sub, "gvkey"])
    res = fit_ols_clustered(y, X, cl)
    r2c, adjc = centered_r2(res, X.shape[1])
    counts = [sum(c.startswith(p) for c in X.columns) for p in ("iy_", "b_", "l_")]

    print(f"    {event_code(event):<4} N={len(y):,}  R²={r2c:.4f}  " + "  ".join(
        f"{v.split('_', 1)[1]}={res.params[v]:.3f}{_stars(res.pvalues[v])}"
        for v in experience if v in res.params.index))

    return ([event_label]
            + [f"{res.params[v]:.3f}{_stars(res.pvalues[v])}" if v in res.params.index else ""
               for v in experience]
            + ["", f"{len(y):,}", f"{r2c:.4f}", f"{adjc:.4f}"] + [str(c) for c in counts])


def build_panel(df, dv, X_fe, fe_bor, fe_lender, fe_labels, sample_suf: str) -> pd.DataFrame:
    cols = {event_code(ev): run_experience_model(df, dv, X_fe, fe_bor, fe_lender, ev, sample_suf)
            for ev in EVENTS}
    index = ["Lender event"] + EXPERIENCE_BASE + ["", "N", "R²", "Adj. R²"] + fe_labels
    return pd.DataFrame(cols, index=index)[[event_code(ev) for ev in EVENTS]]


def run() -> None:
    print("Loading contracts …")
    df_full = pd.read_parquet(DATA_DIR / "fulldata.parquet")
    print(f"  {len(df_full):,} rows × {df_full.shape[1]} cols")

    df, dv, X_fe, fe_bor, fe_lender, fe_labels = prepare_sample(df_full, load_lender_lists())

    panels = []
    for title, sample_suf in LENDER_SAMPLES:
        print(f"\n  {title}")
        panels.append((title, build_panel(df, dv, X_fe, fe_bor, fe_lender, fe_labels, sample_suf)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        row, ws = 0, None
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
