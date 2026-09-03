"""
table6_robust_notop5.py  —  Table 6 robustness: excluding top-5 bank lead arrangers
===================================================================================

Table 6 (RQ3, ASC 842) re-estimated on contracts whose ADMINISTRATIVE AGENT is not one of
the five top-tier lenders. Same four panels, same specification, same FE structure and
standard errors as table6_rq3_asc842.py — the ONLY change is the sample restriction.

    Panel A   ±3-year window   accounting_policy (composite)
    Panel B   ±5-year window   accounting_policy (composite)
    Panel C   ±3-year window   accounting_policy DECOMPOSED into gaap_override + freeze
    Panel D   ±5-year window   accounting_policy DECOMPOSED into gaap_override + freeze

THE EXCLUSION RULE ("D3"). A contract is dropped when any of the five top-tier lenders below
is among the deal's CREDITED LEAD ARRANGERS — DealScan's `lead_arranger` field, resolved to
`lender_parent_id`. The five ids are not a hand-made list: they are exactly the parent ids that
the lender-experience files ever flag `selected_grouping_* == True`, i.e. the same top-5
identity that defines the Top5_* experience variables in Tables 4/5, and the same five banks
Tseng 11-10-25 p.24 names ("Mitsubishi UFJ, Bank of America, JPMorgan, Wells Fargo, Mizuho").

    151226  BofA Securities                      49511  Mitsubishi UFJ Financial Group
     14261  JP Morgan                            48055  Mizuho Financial Group
     19499  Wells Fargo & Co

⚠ SUPERSEDES an earlier rule that flagged a top-5 bank sitting in the `primary_role ==
"Admin agent"` seat. That construct had no basis in the paper and its results are void.
`lead_arranger` is the field the paper itself uses to define the lead-lender pool (p.25,
pp.42-43), so the exclusion now matches the project's own definition of a lead arranger.

Note the FE block is UNCHANGED — this script still uses table6's all-lender FEs. Only the
sample restriction differs from code/manuscript/table6_rq3_asc842.py.

⚠⚠ POWER. The exclusion is expensive, and Panels A/C are reported for completeness rather
than as informative tests. With Borrower FEs, `post_adoption` and its interactions are
identified only off firms observed on BOTH sides of adoption — on the ESTIMATION sample
(the ALL column, after the regressor-completeness filter):

                        N                    identifying firms / obs
                        full      excl.      full             excl.
    ±3y (Panels A/C)    1,704       612      305 /   914      99 / 296
    ±5y (Panels B/D)    2,602       909      444 / 1,573     133 / 457

So roughly 29% of the identifying variation survives, and standard errors widen by very
roughly 1.8×. Read a shrunken coefficient as low power, NOT as evidence against Table 6;
Panels A/C in particular are near the limit of what this FE structure can estimate.

The console prints the same diagnostic per panel, but on the WINDOWED sample before that
completeness filter (so its counts are larger than the table above — 390 → 142 firms at
±3y, 587 → 204 at ±5y). Quote the estimation-sample figures with the table.

Everything else follows table6_rq3_asc842.py exactly: rating BUCKETS interacted with post,
level variables winsorized 1% BEFORE the interactions are formed, non-adopters dropped
rather than coerced to zero, FEs built AFTER the restrictions so the reported counts match
the estimated sample, SEs clustered by gvkey, R²/Adj. R² CENTERED, coefficients + stars only.

Lives in code/exploratory/ — not promoted to the manuscript set. Its baseline counterpart
is code/manuscript/table6_rq3_asc842.py; the two differ only in the sample restriction.

Input : data/fulldata.parquet, data/dealscan_raw.parquet
Output: output/tables/table6_notop5.xlsx  (sheets: Panel A, Panel B, Panel C, Panel D)
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_DIR  = Path(__file__).resolve().parent.parent   # code/ — this script lives in code/exploratory/
DATA_DIR  = REPO_DIR.parent / "data"
OUT_DIR   = REPO_DIR.parent / "output" / "tables"
OUT_FILE  = OUT_DIR / "table6_notop5.xlsx"
CONTRACTS = DATA_DIR / "fulldata.parquet"
MERGE_KEYS = ["borrower_id", "tranche_active_date"]

# Top-5 lenders — the parent ids the experience files flag `selected_grouping_* == True`.
TOP5_PARENT_IDS = {
    151226: "BofA Securities",
     14261: "JP Morgan",
     19499: "Wells Fargo & Co",
     49511: "Mitsubishi UFJ Financial Group Inc",
     48055: "Mizuho Financial Group Inc",
}

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


def _lead_arranger_parents() -> pd.Series:
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


def load_top5_admin_flag() -> pd.Series:
    """Per deal-key: is a top-5 bank among the credited lead arrangers?

    Keys absent from DealScan get no flag; prepare_sample treats those as False (not
    excluded) so the restriction never silently drops rows for a reason other than the one
    being tested.
    """
    leads = _lead_arranger_parents()
    flag = leads.apply(lambda ids: any(i in TOP5_PARENT_IDS for i in ids)).rename("top5_admin")
    print(f"  top-5 lead-arranger flag: {len(flag):,} deal-keys; "
          f"{flag.mean() * 100:.1f}% led by a top-5 bank")
    return flag


def report_identifying_variation(df: pd.DataFrame, label: str) -> None:
    """Firms/observations that actually identify post_adoption under Borrower FEs.

    Reported on the WINDOWED sample, i.e. before each column's regressor-completeness
    filter, so these counts exceed the per-column estimation samples printed below.
    """
    pv    = df.groupby("gvkey")[POST].nunique()
    firms = pv[pv > 1].index
    print(f"    {label:<18} N={len(df):,}  firms={df['gvkey'].nunique():,}  |  "
          f"identifying: {len(firms):,} firms / {int(df['gvkey'].isin(firms).sum()):,} obs")


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

def prepare_sample(lender_lists: pd.Series, top5_flag: pd.Series, window_years: int):
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

    # ── D3 exclusion: drop contracts whose administrative agent is a top-5 bank ──
    # Unmatched keys are NOT excluded (fillna False) — see load_top5_admin_flag.
    excl = df.join(top5_flag, on=MERGE_KEYS)["top5_admin"].fillna(False).astype(bool)
    report_identifying_variation(df, "before exclusion")
    df = df[~excl].copy()
    print(f"  D3 exclusion: dropped {int(excl.sum()):,} contract(s) with a top-5 lead arranger "
          f"({excl.mean() * 100:.1f}%) → {len(df):,} remain "
          f"| post=1: {int((df[POST] == 1).sum()):,}  post=0: {int((df[POST] == 0).sum()):,}")
    report_identifying_variation(df, "after exclusion")

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
    # SEs are echoed here (not in the table, whose format matches table6_rq3_asc842.py)
    # because power, not point estimates, is the thing to judge in this robustness run.
    print(f"    {sheet:<8} N={len(y):,}  R²={r2c:.4f}  clusters={cl.nunique():,}  | " + "  ".join(
        f"{k}={res.params[k]:.3f}{_stars(res.pvalues[k])} (se {res.bse[k]:.3f})"
        for k in key if k in res.params.index))

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
    top5_flag    = load_top5_admin_flag()
    cache, sheets = {}, {}
    for name, yrs, acct in PANELS:
        kind = "composite" if acct == COMPOSITE else "decomposed: " + " + ".join(acct)
        print(f"\n{name}  (±{yrs}y, {kind})")
        if yrs not in cache:                      # windowed sample + FEs reused across panels
            cache[yrs] = prepare_sample(lender_lists, top5_flag, yrs)
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
