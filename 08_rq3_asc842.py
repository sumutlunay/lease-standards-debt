"""
08_rq3_asc842.py — RQ3: does ASC 842 moderate the RQ2 lender-experience associations?

RQ3 asks whether the adoption of ASC 842 changed the way lenders' exposure to irregularity
events in their other borrowers maps into off-balance-sheet lease covenant design.

This script currently produces the FIRST sheet of the RQ3 table (`comparison`): a firm-level
comparison of contracting activity before vs after each firm's ASC 842 adoption date.  It also
builds the loan-level sample and the `post_adoption` treatment dummy that the regression sheets
will use.  The regressions themselves are NOT written yet; `run()` assembles a
{sheet_name: DataFrame} dict so new sheets slot in without touching the writer.

The treatment dummy — post_adoption
-----------------------------------
    1   tranche_active_date >= the borrower's adoption_date   (post-842)
    0   tranche_active_date <  the borrower's adoption_date   (pre-842)
    NaN the firm has no adoption_date

`adoption_date` comes from upstream: 03_contracts.py LEFT-joins asc_adoption_counts.csv on
`cik`, so firms that never adopted (or never matched the ASC file) carry a missing date.  Those
rows are left NaN ON PURPOSE and drop out of any regression including post_adoption — i.e. RQ3
is estimated on ADOPTING FIRMS ONLY.  Coercing them to 0 would silently pool never-adopters
into the pre-period control group and change the estimand.

⚠ NOTE FOR THE REGRESSIONS: Model 7 carries Industry×Year fixed effects, so a post_adoption
MAIN effect is collinear with year and will be absorbed.  RQ3 is identified off the
INTERACTIONS (experience × post_adoption), not the level.

Source
------
ASC adoption:  https://ucdavis.box.com/shared/static/2xb866l3q86x85hc77huzbd2spngdufi.csv
               (asc_adoption_counts.csv — 4,185 firms, one row per CIK)

This is the same file `03_contracts.py` left-joins onto the loan sample on `cik`.  We read it
directly rather than going through contracts.parquet because the counts are FIRM attributes:
in the loan-level sample each firm recurs once per loan (~3.8x), which would inflate n and
shrink the standard errors by roughly a factor of four.  The firm is the correct unit.

Sample
------
ALL 4,185 firms.  Firms with pre = post = 0 are deliberately KEPT: their paired difference is
exactly zero, so they contribute nothing to the t-statistic numerator and are dropped outright
by Wilcoxon.  Excluding them would rescale the means (and inflate Cohen's dz) without adding
a single unit of statistical evidence.  4,185 is the honest denominator.

Tests
-----
Paired (same firm supplies both values), two-sided.  Reported alongside a Wilcoxon signed-rank
test because the counts are zero-inflated and right-skewed (medians of 0).

⚠ OPEN — the window over which pre_*/post_* are counted is not documented.  If it is symmetric
around the adoption date the means are directly comparable and these tests stand as reported.
If the windows are asymmetric (e.g. full sample period: ~18 years pre vs ~4.5 post) the raw
difference is not interpretable and needs exposure-scaling.  The totals argue for symmetry —
post (1,268 contracts) EXCEEDS pre (1,132), whereas a full-period window would make pre ~4x
larger — but this is an inference from the arithmetic, not a definition.  CONFIRM WITH AYUNG.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

REPO_DIR  = Path(__file__).resolve().parent
DATA_DIR  = REPO_DIR.parent / "data"
OUT_DIR   = REPO_DIR.parent / "output" / "tables"
OUT_FILE  = OUT_DIR / "rq3_asc842.xlsx"
CONTRACTS = DATA_DIR / "contracts.parquet"

ASC_SRC = "https://ucdavis.box.com/shared/static/2xb866l3q86x85hc77huzbd2spngdufi.csv"

MERGE_KEYS = ["borrower_id", "tranche_active_date"]

# (pre column, post column, display label)
COMPARISONS = [
    ("pre_contract_count",  "post_contract_count",  "Contracts"),
    ("pre_amendment_count", "post_amendment_count", "Amendments"),
]

# ── Loan-level sample (for the regression sheets) ─────────────────────────────────
# Sample filters and regressor lists are identical to 05 / 07 (Model 7).
DV_SCORES = ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
             "claude_VAR_SCORE", "claude_RES_SCORE"]

# Five dependent variables, one output column each (same construction as 05's five sheets):
# DV = 1 if ANY listed claude_* score > 0, else 0.
DV_SPECS = [
    ("SLB",     ["claude_SLB_SCORE"]),
    ("SYN",     ["claude_SYN_SCORE"]),
    ("OPL",     ["claude_OPL_SCORE"]),
    ("VAR-RES", ["claude_VAR_SCORE", "claude_RES_SCORE"]),
    ("ALL",     DV_SCORES),
]

DETERMINANTS = ["accounting_policy", "offbslease", "num_rating", "non_rated", "relationship_freq"]
CONTROLS = [
    "maturity", "log_lender_count", "log_interest", "log_deal_amount", "perf_pricing",
    "fin_covenant_count", "gen_covenant_count", "is_covenant_ratio", "secured",
    "size", "profitability", "bsfixed", "liabilities", "logage", "btm", "capex",
    "loss", "rand", "divyield",
    "amendment",            # 1 if the contract is an amendment (amendment_seq > 0) — added for RQ3
]
COVENANT_RATIO_FILL = "is_covenant_ratio"

# Non-logged, non-dummy level variables winsorized at 1% both tails on the estimation sample
# (mirrors 05 / 07).  Interactions are formed AFTER winsorizing, so post × offbslease and
# post × fin_covenant_count inherit the winsorized parent — otherwise the interaction would
# smuggle the raw tails back into the model.
WINSOR_LEVEL_VARS = [
    "offbslease",
    "fin_covenant_count", "gen_covenant_count", "is_covenant_ratio",
    "profitability", "bsfixed", "liabilities", "btm", "capex", "rand", "divyield",
]

# ── RQ3A: post_adoption and its interactions ─────────────────────────────────────
POST = "post_adoption"

# Variables interacted with post_adoption.  Order here is the order of the output rows.
INTERACT_VARS = [
    "accounting_policy",
    "relationship_freq",
    "fin_covenant_count",
    "offbslease",
    "non_rated",
    "num_rating",
    "amendment",
]


def interaction_name(var: str) -> str:
    return f"{POST}_x_{var}"


INTERACTIONS = [interaction_name(v) for v in INTERACT_VARS]
TEST_VARS    = [POST] + INTERACTIONS


def stars(p: float) -> str:
    """Significance stars at the .10 / .05 / .01 levels (same convention as 05–07)."""
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def compare(df: pd.DataFrame, pre: str, post: str) -> dict:
    """Paired pre-vs-post comparison for one count pair, on all firms with both values."""
    sub  = df.dropna(subset=[pre, post])
    a    = sub[pre].astype(float)
    b    = sub[post].astype(float)
    diff = b - a

    t_stat, p_t = stats.ttest_rel(b, a)

    # Wilcoxon drops zero differences (zero_method="wilcox"); guard the all-tied case.
    if (diff != 0).any():
        _, p_w = stats.wilcoxon(b, a, zero_method="wilcox")
    else:
        p_w = np.nan

    sd_diff = diff.std(ddof=1)
    dz      = diff.mean() / sd_diff if sd_diff > 0 else np.nan

    return {
        "Pre-adoption mean":      f"{a.mean():.3f}",
        "  (SD)":                 f"({a.std():.3f})",
        "  Median":               f"{a.median():.1f}",
        "Post-adoption mean":     f"{b.mean():.3f}",
        "  (SD) ":                f"({b.std():.3f})",   # trailing space keeps the label unique
        "  Median ":              f"{b.median():.1f}",
        "Difference (post − pre)": f"{diff.mean():.3f}{stars(p_t)}",
        "  (SD of difference)":   f"({sd_diff:.3f})",
        "Paired t-statistic":     f"{t_stat:.3f}",
        "  p-value":              f"{p_t:.4f}",
        "Wilcoxon signed-rank p": f"{p_w:.4f}",
        "Cohen's dz":             f"{dz:.3f}",
        "":                       "",
        "N firms":                f"{len(sub):,}",
        "  Firms post > pre":     f"{(diff > 0).sum():,}",
        "  Firms post < pre":     f"{(diff < 0).sum():,}",
        "  Firms tied":           f"{(diff == 0).sum():,}",
    }


def build_comparison(asc: pd.DataFrame) -> pd.DataFrame:
    """The `comparison` sheet: one column per count pair."""
    cols = {label: compare(asc, pre, post) for pre, post, label in COMPARISONS}
    tab  = pd.DataFrame(cols)

    header = pd.DataFrame(
        {label: ["Firm-level, all firms"] for _, _, label in COMPARISONS},
        index=["Sample"],
    )
    return pd.concat([header, tab])


# ── Fixed-effect builders (dense; copied from 05 / 07) ───────────────────────────

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


def load_lender_lists() -> pd.Series:
    """(borrower_id, tranche_active_date) → [lender_parent_id, …], for the lender FEs."""
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


# ── Estimation helpers (copied from 05 / 07) ─────────────────────────────────────

def stabilize_design(X: pd.DataFrame, y: pd.Series, clusters: pd.Series):
    """Remove non-finite rows, constant columns, duplicate columns, singleton FE columns."""
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

    print(f"    stabilize: {n_dropped} non-finite rows dropped, {X.shape[1]} cols remaining")
    return X.astype(float), y, clusters


def fit_ols_clustered(y: pd.Series, X: pd.DataFrame, clusters: pd.Series):
    model = sm.OLS(y, X, hasconst=False)
    return model.fit(cov_type="cluster",
                     cov_kwds={"groups": clusters, "use_correction": True})


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
    """Winsorize each column at [p, 1-p], bounds computed on the regression sample."""
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


# ── Loan-level sample + the post_adoption treatment dummy ────────────────────────

def prepare_sample(lender_lists: pd.Series):
    """Loan-level RQ3 sample (same filters as 05/07) restricted to ADOPTING FIRMS, with the
    five DV dummies and `post_adoption` attached, plus the dense FE matrices.

    post_adoption = 1 if the tranche was signed on/after the borrower's ASC 842 adoption date
                    0 if signed before
                  NaN if the firm has no adoption_date

    `adoption_date` arrives upstream: 03_contracts.py LEFT-joins asc_adoption_counts.csv on
    `cik`, so non-adopters (and firms that never matched the ASC file) carry a missing date.
    Those rows are DROPPED here — not coerced to 0 — so RQ3 is estimated on adopting firms
    only.  A 0 would silently pool never-adopters into the pre-period control group.

    The FE matrices are built AFTER that restriction, so the Industry×Year / Borrower / Lender
    FE counts reflect the sample actually estimated.
    """
    df = pd.read_parquet(CONTRACTS)
    print(f"\nLoaded {CONTRACTS.name}: {len(df):,} rows × {df.shape[1]} cols")

    # Sample filters (identical to 05 / 07)
    df = df[df["claude_is_debt_contract"] == "Y"].copy()
    print(f"  {len(df):,} rows after keeping claude_is_debt_contract == Y")
    df = df[df[DV_SCORES].notna().all(axis=1)].copy()
    print(f"  {len(df):,} rows after dropping rows missing any claude score")

    # Five DVs — all five raw scores are jointly present, so the sample is identical across them
    for sheet, cols in DV_SPECS:
        df[f"{sheet}_score_dummy"] = (df[cols] > 0).any(axis=1).astype(float)

    df["contract_year"] = df["tranche_active_date"].dt.year

    gaap   = df["claude_GAAP_OVERRIDE_SCORE"]
    freeze = df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan
    df[COVENANT_RATIO_FILL] = df[COVENANT_RATIO_FILL].fillna(0)

    # ── post_adoption ────────────────────────────────────────────────────────────
    adopt = pd.to_datetime(df["adoption_date"], errors="coerce")
    df[POST] = np.where(adopt.isna(), np.nan,
                        (df["tranche_active_date"] >= adopt).astype(float))

    n_all = len(df)
    print(f"\n── post_adoption ─────────────────────────────────────────────────────")
    print(f"  adopting firms (post_adoption observed) : {df[POST].notna().sum():,} / {n_all:,}")
    print(f"  non-adopters / unmatched (NaN, DROPPED) : {df[POST].isna().sum():,}")

    # RESTRICT to adopting firms
    df = df[df[POST].notna()].copy()
    print(f"\n  Restricted to adopting firms: {len(df):,} rows")
    print(f"    post_adoption = 1 (post-842) : {int((df[POST] == 1).sum()):,}")
    print(f"    post_adoption = 0 (pre-842)  : {int((df[POST] == 0).sum()):,}")

    for sheet, _ in DV_SPECS:
        d = df[f"{sheet}_score_dummy"]
        print(f"    {sheet + '_score_dummy':<22} mean = {d.mean():.3f}")

    # ── Fixed effects (built on the RESTRICTED sample) ───────────────────────────
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
    return df, X_fe, fe_bor, fe_lender, fe_labels


# ── RQ3A estimation ──────────────────────────────────────────────────────────────

def run_rq3a_column(df, X_fe, fe_bor, fe_lender, fe_labels, sheet: str, col_name: str) -> list:
    """Estimate one RQ3A column (one dependent variable) and return the formatted column."""
    dv         = f"{sheet}_score_dummy"
    regressors = TEST_VARS + DETERMINANTS + CONTROLS
    IY_LABEL, BOR_LABEL, LEN_LABEL = fe_labels

    print(f"\n{'=' * 60}\n  RQ3A {col_name}: DV = {dv}\n{'=' * 60}")

    df = df.copy()

    # Sample mask on the PARENT variables (the interactions are deterministic functions of
    # them, so they add no further missingness).
    parents = [POST] + DETERMINANTS + CONTROLS
    finite  = np.isfinite(df[parents].to_numpy(dtype=float)).all(axis=1)
    mask    = (pd.Series(finite, index=df.index)
               & df[dv].notna() & df["gvkey"].notna())
    print(f"  Estimation sample (all regressors non-missing): {int(mask.sum()):,}")

    # Winsorize FIRST, then form the interactions, so post × offbslease and
    # post × fin_covenant_count inherit the winsorized parent.
    print("  winsorizing level variables at 1% on this sample:")
    dfw = winsorize_cols(df, WINSOR_LEVEL_VARS, mask)
    for v in INTERACT_VARS:
        dfw[interaction_name(v)] = dfw[POST] * dfw[v]

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

    dropped = [v for v in TEST_VARS if v not in X.columns]
    if dropped:
        print(f"  ⚠ dropped by stabilize (collinear/constant): {', '.join(dropped)}")
    for v in TEST_VARS:
        if v in res.params.index:
            print(f"    {v:<34} {res.params[v]:>8.4f}{_stars(res.pvalues[v]):<3} "
                  f"(t={res.tvalues[v]:.3f})")

    values = [dv]
    for var in regressors:
        if var in res.params.index:
            values.append(f"{res.params[var]:.3f}{_stars(res.pvalues[var])}")
            values.append(f"({res.tvalues[var]:.3f})")
        else:
            values.append("")
            values.append("")

    values.append("")
    values.append(f"{len(y_clean):,}")
    values.append(f"{res.rsquared:.4f}")
    values.append(f"{res.rsquared_adj:.4f}")
    for label in fe_labels:
        values.append(str(fe_counts.get(label, "")))

    return values


def build_rq3a(df, X_fe, fe_bor, fe_lender, fe_labels) -> pd.DataFrame:
    """The `RQ3A` sheet: five columns, one per dependent variable."""
    col_data = {}
    for i, (sheet, _) in enumerate(DV_SPECS, start=1):
        col_name = f"({i})"
        col_data[col_name] = run_rq3a_column(
            df, X_fe, fe_bor, fe_lender, fe_labels, sheet, col_name,
        )

    display = [POST] + [f"{POST} × {v}" for v in INTERACT_VARS] + DETERMINANTS + CONTROLS
    index   = (["Dependent variable"]
               + _build_labels(display)
               + ["", "N", "R²", "Adj. R²"] + fe_labels)
    return pd.DataFrame(col_data, index=index)


def run() -> None:
    print(f"\n{'#' * 60}\n#  RQ3 — ASC 842\n{'#' * 60}")

    print("\nLoading ASC adoption counts …")
    asc = pd.read_csv(ASC_SRC)
    print(f"  {len(asc):,} rows × {asc.shape[1]} cols   ({asc['cik'].nunique():,} unique CIK)")

    if asc["cik"].nunique() != len(asc):
        raise ValueError("asc_adoption_counts.csv is not unique on cik — the paired tests "
                         "assume one row per firm.")

    adopt_yr = pd.to_datetime(asc["adoption_date"], errors="coerce").dt.year
    print("\nAdoption year distribution:")
    print(adopt_yr.value_counts().sort_index().to_string())

    print("\n── Pre vs post comparison (paired, all firms) ────────────────────────")
    for pre, post, label in COMPARISONS:
        r    = compare(asc, pre, post)
        diff = r["Difference (post − pre)"]
        dz   = r["Cohen's dz"]
        print(f"  {label:<11s} pre={r['Pre-adoption mean']:>6s}  post={r['Post-adoption mean']:>6s}"
              f"  diff={diff:>9s}  t={r['Paired t-statistic']:>7s}"
              f"  p={r['  p-value']}  dz={dz}")

    sheets = {"comparison": build_comparison(asc)}

    # ── RQ3A: the five-DV table with post_adoption and its interactions ──────────
    print(f"\n{'#' * 60}\n#  RQ3A — post_adoption × determinants  (worksheet 'RQ3A')\n{'#' * 60}")
    lender_lists = load_lender_lists()
    df, X_fe, fe_bor, fe_lender, fe_labels = prepare_sample(lender_lists)
    sheets["RQ3A"] = build_rq3a(df, X_fe, fe_bor, fe_lender, fe_labels)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        for name, tab in sheets.items():
            tab.to_excel(xw, sheet_name=name)
            ws = xw.sheets[name]
            ws.set_column(0, 0, 30)
            ws.set_column(1, tab.shape[1], 16)

    print(f"\nSaved → {OUT_FILE}  (sheets: {', '.join(sheets)})")


if __name__ == "__main__":
    run()
