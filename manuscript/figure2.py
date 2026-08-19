"""
figure2.py  —  Figure 2: RQ3 event study (accounting_policy × event time)
=========================================================================

Table 6 summarises the post-842 shift in the accounting-policy / off-BS-lease-covenant
association with a single step, `post_adoption × accounting_policy`. This figure GENERALISES
that step into event-time dummies interacted with accounting_policy, and plots the
coefficients ± 95% CI — the standard dynamic-DiD picture.

    post_adoption  and  post_adoption × accounting_policy        (one step)

is replaced by

    accounting_policy × 1(event_time = τ)   for every τ          (dynamic)

    event_time = contract_year − the firm's ASC 842 adoption_year
    reference  = τ = 0 (the adoption year; its coefficient is pinned to 0)

The event-time MAIN dummies are kept so each interaction is a proper within-τ difference
(most are absorbed by the Industry×Year FE for the 2019 cohort; the rank guard drops the
collinear ones). accounting_policy stays as a main effect — the association at the reference
period. The other RQ3 post-interactions (relationship_freq, offbslease, …) are dropped: this
figure isolates the accounting_policy dynamic. `amendment` remains a plain control.

★ WINDOW. Estimated on the **±3-year** window (N = 1,704) so the figure matches **Table 6
Panel A** row-for-row. The ±5-year variant lives in exploratory/figures_rq3_eventstudy.py
and matches Panel B. Event time therefore spans τ ∈ [−3, +3] by construction, and — unlike
the ±5-year version, whose τ = +5 bin held just 4 contracts and had to be pooled into a
"≥ +4" endpoint — every bin here is a genuine single-year bin. The thinnest, τ = +3, holds
57 contracts (32 with accounting_policy = 1), so its CI is wider than the others; bins run
195–315 elsewhere.

Credit quality enters as the BUCKET dummies (BB_grade / B_grade / CCC_below /
non_rated_suppl_all, with ig_grade the omitted reference) as plain level controls — Table 6's
post × bucket interactions have no analogue here. Everything else follows the Model 7 spec:
same FE, determinants, controls and winsorisation.

Adoption is a near-single 2019 event (82.5% of firms), so event time ≈ calendar year − 2019
for most of the sample.

Input : data/fulldata.parquet, data/dealscan_raw.parquet
Output: output/figures/figure2.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

REPO_DIR  = Path(__file__).resolve().parent.parent  # code/ — this script lives in code/manuscript/
DATA_DIR  = REPO_DIR.parent / "data"
FIG_DIR   = REPO_DIR.parent / "output" / "figures"
OUT_FILE  = FIG_DIR / "figure2.png"
CONTRACTS = DATA_DIR / "fulldata.parquet"

MERGE_KEYS = ["borrower_id", "tranche_active_date"]

# ── Event-time window ─────────────────────────────────────────────────────────────
WINDOW_YEARS       = 3          # date-based ±WINDOW_YEARS window — matches Table 6 Panel A
# The window guarantees τ ∈ [−3, +3] and every bin is populated (thinnest is τ = +3 at 57
# contracts), so no endpoint pooling is needed here — unlike the ±5-year variant.
ETIME_LO, ETIME_HI = -3, 3
REF_TAU            = 0          # reference period (coefficient pinned to 0) — the adoption year

# ── Model 7 regressor lists (identical to 08_rq3_asc842.py) ───────────────────────
DV_SCORES = ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
             "claude_VAR_SCORE", "claude_RES_SCORE"]
DV = "ALL_score_dummy"          # headline DV: 1 if ANY claude_* score > 0

# accounting_policy is handled separately (main effect + event-time interactions), so it is
# NOT in this list; the other determinants enter as plain levels.
# Credit quality uses the 08d BUCKET spec: ig_grade (investment grade) is the OMITTED reference,
# so it is not a regressor; BB_grade / B_grade / CCC_below / non_rated_suppl_all are included.
RATING_BUCKETS = ["BB_grade", "B_grade", "CCC_below", "non_rated_suppl_all"]
DETERMINANTS = ["offbslease"] + RATING_BUCKETS + ["relationship_freq"]
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


# ── FE builders (copied from 08) ──────────────────────────────────────────────────

def make_industry_year_fe(df: pd.DataFrame, sic_digits: int = 2):
    sic_str  = df["sic"].fillna(0).astype(int).astype(str).str.zfill(4).str[:sic_digits]
    ind_year = sic_str + "_" + df["contract_year"].astype(str)
    fe       = pd.get_dummies(ind_year, prefix="iy", drop_first=False, dtype=float)
    return fe, int(ind_year.nunique())


def make_borrower_fe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df["borrower_id"].astype("Int64"), prefix="b",
                          drop_first=False, dtype=float)


def make_lender_multi_hot(df: pd.DataFrame, lender_lists: pd.Series) -> pd.DataFrame:
    unique_lenders = sorted({lid for lst in lender_lists for lid in lst})
    if not unique_lenders:
        return pd.DataFrame(index=df.index)
    data = {
        f"l_{int(lid)}": lender_lists.apply(lambda lst, v=lid: 1.0 if v in lst else 0.0)
        for lid in unique_lenders
    }
    return pd.DataFrame(data, index=df.index)


def load_lender_lists() -> pd.Series:
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


# ── Estimation helpers (copied from 08) ───────────────────────────────────────────

def drop_dependent_columns(X: pd.DataFrame) -> pd.DataFrame:
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


def stabilize_design(X: pd.DataFrame, y: pd.Series, clusters: pd.Series):
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
    X = drop_dependent_columns(X)

    print(f"    stabilize: {n_dropped} non-finite rows dropped, {X.shape[1]} cols remaining")
    return X.astype(float), y, clusters


def fit_ols_clustered(y: pd.Series, X: pd.DataFrame, clusters: pd.Series):
    model = sm.OLS(y, X, hasconst=False)
    return model.fit(cov_type="cluster",
                     cov_kwds={"groups": clusters, "use_correction": True})


def winsorize_cols(frame: pd.DataFrame, cols: list, mask: pd.Series, p: float = 0.01):
    out = frame.copy()
    for c in cols:
        if c not in out.columns:
            continue
        # Cast to float first: some level vars (e.g. covenant counts) are nullable Int64, and
        # clipping them to a fractional quantile bound raises on Int64 (surfaces on the small
        # ±3-year window sample). Estimate-preserving — these regressors are floated downstream.
        out[c] = out[c].astype(float)
        lo = out.loc[mask, c].quantile(p)
        hi = out.loc[mask, c].quantile(1 - p)
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out


# ── Sample (identical restriction to 08's prepare_sample) ─────────────────────────

def prepare_sample(lender_lists: pd.Series):
    df = pd.read_parquet(CONTRACTS)
    print(f"\nLoaded {CONTRACTS.name}: {len(df):,} rows × {df.shape[1]} cols")

    df = df[df["claude_is_debt_contract"] == "Y"].copy()
    df = df[df[DV_SCORES].notna().all(axis=1)].copy()
    print(f"  {len(df):,} rows after sample filters")

    df[DV] = (df[DV_SCORES] > 0).any(axis=1).astype(float)
    df["contract_year"] = df["tranche_active_date"].dt.year

    gaap   = df["claude_GAAP_OVERRIDE_SCORE"]
    freeze = df["claude_FREEZE_SCORE"]
    df["accounting_policy"] = ((gaap > 0) | (freeze > 0)).astype(float)
    df.loc[gaap.isna() & freeze.isna(), "accounting_policy"] = np.nan

    # Restrict to adopting firms (adoption_date present)
    adopt = pd.to_datetime(df["adoption_date"], errors="coerce")
    df = df[adopt.notna()].copy()
    adopt = pd.to_datetime(df["adoption_date"], errors="coerce")
    n_adopt = len(df)
    print(f"  Restricted to adopting firms: {n_adopt:,} rows")

    # ── ±WINDOW_YEARS date-based window around adoption_date (identical to 08b) ──────
    lo = adopt - pd.DateOffset(years=WINDOW_YEARS)
    hi = adopt + pd.DateOffset(years=WINDOW_YEARS)
    in_window = (df["tranche_active_date"] >= lo) & (df["tranche_active_date"] <= hi)
    df = df[in_window].copy()
    adopt = pd.to_datetime(df["adoption_date"], errors="coerce")
    print(f"  ±{WINDOW_YEARS}-year window around adoption_date: {len(df):,} rows "
          f"({n_adopt - len(df):,} dropped)")

    # Event time (years) relative to the firm's adoption year.  The date window puts values in
    # [−WINDOW_YEARS, +WINDOW_YEARS]; the clip pools the thin τ = +5 tail into the ETIME_HI bin.
    df["event_time"] = (df["contract_year"] - adopt.dt.year).clip(ETIME_LO, ETIME_HI).astype(int)

    # ── Fixed effects (built on the restricted sample) ───────────────────────────
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
    fe_lender = make_lender_multi_hot(df, df["lender_ids"])
    print(f"Lender FE matrix: {fe_lender.shape[1]} columns")

    return df, X_fe, fe_bor, fe_lender


# ── Event-time design + estimation ────────────────────────────────────────────────

def build_event_dummies(df: pd.DataFrame):
    """Event-time main dummies and accounting_policy × event-time interactions.

    The reference period (τ = REF_TAU) is omitted from BOTH sets, so every reported coefficient
    is a contrast against the year before adoption.
    """
    taus = [t for t in range(ETIME_LO, ETIME_HI + 1) if t != REF_TAU]
    main, inter = {}, {}
    for t in taus:
        d = (df["event_time"] == t).astype(float)
        main[f"et_{t}"]  = d                       # event-time main effect
        inter[f"ap_x_et_{t}"] = df["accounting_policy"] * d   # the plotted coefficients
    return pd.DataFrame(main, index=df.index), pd.DataFrame(inter, index=df.index), taus


def estimate(df, X_fe, fe_bor, fe_lender):
    # Sample mask on the parent variables (same construction as 08): all determinants/controls
    # non-missing, DV and gvkey present.  accounting_policy must be present too.
    parents = ["accounting_policy"] + DETERMINANTS + CONTROLS
    finite  = np.isfinite(df[parents].to_numpy(dtype=float)).all(axis=1)
    mask    = pd.Series(finite, index=df.index) & df[DV].notna() & df["gvkey"].notna()
    print(f"\n  Estimation sample (all regressors non-missing): {int(mask.sum()):,}")

    # Winsorize the level controls on the estimation sample (before forming anything downstream)
    dfw = winsorize_cols(df, WINSOR_LEVEL_VARS, mask)

    et_main, et_inter, taus = build_event_dummies(dfw)

    # Design: interactions of interest first (so the rank guard never drops them), then
    # accounting_policy main, the other determinants/controls, event-time mains, then FE.
    core = pd.concat(
        [et_inter, dfw[["accounting_policy"]].astype(float),
         dfw[DETERMINANTS + CONTROLS].astype(float), et_main],
        axis=1,
    )
    X_full = pd.concat([core, X_fe, fe_bor, fe_lender], axis=1)

    # Apply the sample mask before stabilising (stabilize also drops non-finite rows, but we
    # restrict to the intended estimation sample explicitly first).
    X_full = X_full.loc[mask]
    y      = dfw.loc[mask, DV]
    cl     = dfw.loc[mask, "gvkey"]

    X, y, cl = stabilize_design(X_full, y, cl)
    res = fit_ols_clustered(y, X, cl)

    print(f"  N = {len(y):,}  |  R² = {res.rsquared:.4f}  |  clusters = {cl.nunique():,}")

    # Collect the accounting_policy × τ coefficients (reference pinned to 0, no CI)
    rows = []
    for t in range(ETIME_LO, ETIME_HI + 1):
        if t == REF_TAU:
            rows.append((t, 0.0, 0.0, True))
            continue
        name = f"ap_x_et_{t}"
        if name in res.params.index:
            rows.append((t, res.params[name], res.bse[name], False))
        else:
            rows.append((t, np.nan, np.nan, False))
    coef = pd.DataFrame(rows, columns=["tau", "beta", "se", "is_ref"])

    print("\n  accounting_policy × event-time coefficients:")
    for _, r in coef.iterrows():
        tag = "  (ref)" if r.is_ref else ""
        print(f"    τ={int(r.tau):>3}   β={r.beta:>8.4f}   SE={r.se:>7.4f}{tag}")

    return coef, len(y)


# ── Plot ──────────────────────────────────────────────────────────────────────────

def plot(coef: pd.DataFrame, n: int):
    z = 1.96                       # 95% CI
    coef = coef.copy()
    coef["lo"] = coef["beta"] - z * coef["se"]
    coef["hi"] = coef["beta"] + z * coef["se"]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    non_ref = coef[~coef["is_ref"]]
    ref     = coef[coef["is_ref"]]

    # Point estimates with 95% CI whisker caps (black, matching the example's style)
    ax.errorbar(non_ref["tau"], non_ref["beta"],
                yerr=z * non_ref["se"],
                fmt="o", color="black", ecolor="black",
                elinewidth=1.0, capsize=3, markersize=4, zorder=3)
    # Reference period: open marker pinned at 0
    ax.plot(ref["tau"], ref["beta"], marker="o", markersize=4,
            markerfacecolor="white", markeredgecolor="black", zorder=4)

    ax.axhline(0, color="black", linewidth=0.9)
    # Adoption year / reference period (τ = 0): mark the line directly on the point.
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)

    ax.set_xlabel("Event time to ASC 842 adoption (years)")
    ax.set_ylabel("Estimate and 95% Conf. Int.")
    ax.set_title("Accounting policy × OBSLI")

    ticks = list(range(ETIME_LO, ETIME_HI + 1))
    ax.set_xticks(ticks)
    # The top bin is a "≥" catch-all whenever it pools years beyond ETIME_HI (see prepare_sample).
    ax.set_xticklabels([f"≥{t}" if t == ETIME_HI < WINDOW_YEARS else str(t) for t in ticks])
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.margins(x=0.03)

    fig.text(0.99, 0.01,
             f"Adopting firms, ±{WINDOW_YEARS}-year window, N = {n:,}. Ref period τ = {REF_TAU}. "
             "Model 7 FE (Industry×Year + Borrower + Lender), SE clustered by gvkey.",
             ha="right", va="bottom", fontsize=6.5, color="0.35")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=200)
    print(f"\nSaved → {OUT_FILE}")


def run():
    print("⚠  REMINDER: R² printed below is UNCENTERED (hasconst=False). 06/06b/06c switched to "
          "CENTERED Adj. R² on 2026-08-11 — update this script to match before using its R².")
    print(f"\n{'#' * 60}\n#  RQ3 event-study — accounting_policy × event time (ALL)\n{'#' * 60}")
    lender_lists = load_lender_lists()
    df, X_fe, fe_bor, fe_lender = prepare_sample(lender_lists)
    coef, n = estimate(df, X_fe, fe_bor, fe_lender)
    plot(coef, n)


if __name__ == "__main__":
    run()
