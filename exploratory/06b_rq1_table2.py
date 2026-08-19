"""
06b_rq1_fe_only.py
RQ1, FE-only variant of 06 — how much of each off-balance-sheet lease-covenant dummy is
explained by the fixed-effect structure ALONE, with NO regressors on the right-hand side.

This is the companion to 06_rq1_determinants.py. Same five dependent variables (one sheet
each) and the SAME eight fixed-effect structures, but every column is a pure projection of
the DV onto its fixed effects — no determinants, no controls. There is nothing to estimate a
coefficient/standard error for, so each column reports only N, R², Adj. R² and the FE counts.
Read it as a variance-decomposition: how far each FE definition (all lenders vs. lead
arrangers vs. a single lead-left, with/without Borrower) goes toward explaining the DV before
any covariate is added.

  (1) Industry×Year
  (2) Industry×Year + Borrower
  (3) Industry×Year + Lender (all syndicate lenders, multi-hot on lender_parent_id)
  (4) Industry×Year + Lead-arranger (all lead arrangers, multi-hot on parsed names)
  (5) Industry×Year + Lead-left (single main arranger per deal)
  (6) Industry×Year + Borrower + Lender (multi-hot)
  (7) Industry×Year + Borrower + Lead-arranger (multi-hot)
  (8) Industry×Year + Borrower + Lead-left

Sample — the FULL scored-contract sample (~14,584): claude_is_debt_contract == Y and all five
claude_*_SCORE present. Because there are no regressors, the regressor-completeness filter that
trims 06 to its m7 sample (N = 11,184) does NOT apply here, so N is larger (and NOT comparable
to 06's R²). N is identical across the eight columns (the FE dummies are always finite, so no
rows drop). 06 itself is left completely intact; this is a standalone sibling script.

R² is CENTERED (1 − SSR/SST about the mean); Adj. R² penalizes by k = the identified FE rank
(after the same singleton/duplicate/linear-dependence guard 06 uses, so the FE counts line up
with 06's post-rank convention). Fixed effects are fit with no separate intercept — the complete
Industry×Year dummy block already spans the constant.

Input:  data/fulldata.parquet       (output of 03_contracts.py + 04_merge.py)
        data/dealscan_raw.parquet    (lender_parent_id + lead_arranger / lead_left fields)
Output: output/tables/table2.xlsx  (one sheet per DV: SLB, SYN, …)
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent  # code/ — this script lives in code/exploratory/
DATA_DIR = REPO_DIR.parent / "data"
OUT_DIR  = REPO_DIR.parent / "output" / "tables"
OUT_FILE = OUT_DIR / "table2.xlsx"

DV_SPECS = [
    ("SLB",     ["claude_SLB_SCORE"]),
    ("SYN",     ["claude_SYN_SCORE"]),
    ("OPL",     ["claude_OPL_SCORE"]),
    ("VAR-RES", ["claude_VAR_SCORE", "claude_RES_SCORE"]),
    ("ALL",     ["claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
                 "claude_VAR_SCORE", "claude_RES_SCORE"]),
]
MERGE_KEYS = ["borrower_id", "tranche_active_date"]


# ── DealScan lead-arranger / lead-left parsing (identical to 06) ──────────────────
_SHARE_RE = re.compile(r"\s*\d{1,3}(?:\.\d+)?\s*%")


def parse_arrangers(s) -> list:
    if pd.isna(s) or str(s).strip() in ("", "None"):
        return []
    cleaned = _SHARE_RE.sub("", str(s))
    seen, out = set(), []
    for tok in cleaned.split(","):
        t = tok.strip()
        if t and t != "None" and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def clean_one_name(s):
    if pd.isna(s) or str(s).strip() in ("", "None"):
        return None
    t = _SHARE_RE.sub("", str(s)).split(",")[0].strip()
    return t or None


# ── Fixed-effect builders (identical to 06) ──────────────────────────────────────

def make_industry_year_fe(df: pd.DataFrame, sic_digits: int = 2):
    sic_str  = df["sic"].fillna(0).astype(int).astype(str).str.zfill(4).str[:sic_digits]
    ind_year = sic_str + "_" + df["contract_year"].astype(str)
    fe       = pd.get_dummies(ind_year, prefix="iy", drop_first=False, dtype=float)
    return fe, int(ind_year.nunique())


def make_borrower_fe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df["borrower_id"].astype("Int64"), prefix="b",
                          drop_first=False, dtype=float)


def make_lender_multi_hot(df: pd.DataFrame, lender_lists: pd.Series) -> pd.DataFrame:
    s = lender_lists.explode().dropna()
    if s.empty:
        return pd.DataFrame(index=df.index)
    s  = s.astype(int)
    oh = pd.get_dummies(s, prefix="l", prefix_sep="_", dtype=float).groupby(level=0).max()
    return oh.reindex(df.index, fill_value=0.0)


def make_name_multi_hot(name_lists: pd.Series, prefix: str) -> pd.DataFrame:
    s = name_lists.explode().dropna()
    if s.empty:
        return pd.DataFrame(index=name_lists.index)
    idmap = {n: i for i, n in enumerate(sorted(s.unique()))}
    oh = (pd.get_dummies(s.map(idmap), prefix=prefix.rstrip("_"), prefix_sep="_", dtype=float)
          .groupby(level=0).max())
    return oh.reindex(name_lists.index, fill_value=0.0)


def make_single_cat_fe(names: pd.Series, prefix: str) -> pd.DataFrame:
    codes = names.fillna("__NONE__").astype(str)
    idmap = {n: i for i, n in enumerate(sorted(codes.unique()))}
    return pd.get_dummies(codes.map(idmap), prefix=prefix.rstrip("_"),
                          prefix_sep="_", dtype=float)


# ── Shared inputs (identical to 06) ──────────────────────────────────────────────

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


def load_lead_arranger_info() -> pd.DataFrame:
    print("\nLoading lead-arranger / lead-left fields from dealscan raw …")
    ds = pd.read_parquet(
        DATA_DIR / "dealscan_raw.parquet",
        columns=["borrower_id", "tranche_active_date", "lead_arranger", "lead_left"],
    )
    ds["tranche_active_date"] = pd.to_datetime(ds["tranche_active_date"], errors="coerce")
    ds = ds.dropna(subset=MERGE_KEYS)
    for c in ["lead_arranger", "lead_left"]:
        ds[c] = ds[c].where(ds[c].astype(str).str.strip().ne("None"))
    g = ds.groupby(MERGE_KEYS)[["lead_arranger", "lead_left"]].first()

    g["arr_names"] = g["lead_arranger"].apply(parse_arrangers)
    g["ll_native"] = g["lead_left"].apply(clean_one_name)
    g["lead_left_name"] = g.apply(
        lambda r: r["ll_native"] if r["ll_native"]
        else (r["arr_names"][0] if len(r["arr_names"]) == 1 else "__NONE__"),
        axis=1,
    )
    cov_ll = (g["lead_left_name"] != "__NONE__").mean()
    print(f"  {len(g):,} deal-keys; lead-left assigned on {cov_ll*100:.1f}%")
    return g[["arr_names", "lead_left_name"]]


# ── Design cleaner + FE-only fit ─────────────────────────────────────────────────

def drop_dependent_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are exact linear combinations of columns to their LEFT (unpivoted QR),
    so the identified FE rank is well-defined. Estimate-preserving (span unchanged)."""
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


def stabilize_fe(X: pd.DataFrame, y: pd.Series):
    """FE-only cleaner: drop non-finite rows, constant/duplicate/singleton FE columns, then
    linearly-dependent columns. No clusters (there are no coefficients/SEs to compute)."""
    row_ok = (np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
              & np.isfinite(y.to_numpy(dtype=float)))
    n_dropped = int((~row_ok).sum())
    X, y = X.loc[row_ok], y.loc[row_ok]

    X = X.loc[:, X.std(ddof=0) > 0]
    X = X.loc[:, ~X.T.duplicated()]
    X = X.loc[:, X.sum(axis=0) != 1]
    X = drop_dependent_columns(X)

    print(f"    stabilize: {n_dropped} non-finite rows dropped, {X.shape[1]} cols remaining")
    return X.astype(float), y


def fe_only_fit(X: pd.DataFrame, y: pd.Series):
    """Project y onto the (full-rank) FE design and return (r2_centered, adj_r2, n, k)."""
    Xv, yv = X.to_numpy(dtype=float), y.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid = yv - Xv @ beta
    ss_res = float(resid @ resid)
    yc     = yv - yv.mean()
    ss_tot = float(yc @ yc)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
    n, k = Xv.shape
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else float("nan")
    return r2, adj, n, k


# ── Per-DV table builder ──────────────────────────────────────────────────────────

def build_table(df_full: pd.DataFrame, lender_lists: pd.Series, lead_info: pd.DataFrame,
                sheet_name: str, score_cols: list) -> pd.DataFrame:
    dv = f"{sheet_name}_score_dummy"
    print(f"\n{'#' * 60}\n#  {sheet_name}:  {dv}  (FE-only)\n{'#' * 60}")

    df = df_full[df_full["claude_is_debt_contract"] == "Y"].copy()
    df = df[df[score_cols].notna().all(axis=1)].copy()
    print(f"  {len(df):,} rows (full scored-contract sample)")

    df[dv]              = (df[score_cols] > 0).any(axis=1).astype(float)
    df["contract_year"] = df["tranche_active_date"].dt.year
    print(f"  {dv} distribution: {df[dv].value_counts().sort_index().to_dict()}")

    # ── Fixed-effect matrices ────────────────────────────────────────────────
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

    df = df.join(lead_info, on=MERGE_KEYS)
    df["arr_names"]      = df["arr_names"].apply(lambda x: x if isinstance(x, list) else [])
    df["lead_left_name"] = df["lead_left_name"].where(df["lead_left_name"].notna(), "__NONE__")
    fe_leadarr  = make_name_multi_hot(df["arr_names"], prefix="la_")
    fe_leadleft = make_single_cat_fe(df["lead_left_name"], prefix="ll_")
    print(f"Lead-arranger FE matrix: {fe_leadarr.shape[1]} columns")
    print(f"Lead-left FE matrix: {fe_leadleft.shape[1]} columns")

    y = df[dv]

    IY_LABEL, BOR_LABEL = f"Industry×Year FEs (SIC {sic_digits}-digit)", "Borrower FEs"
    LEN_LABEL           = "Lender FEs"
    LA_LABEL, LL_LABEL  = "Lead-arranger FEs", "Lead-left FEs"
    fe_label_order = [IY_LABEL, BOR_LABEL, LEN_LABEL, LA_LABEL, LL_LABEL]

    col_specs = {
        "(1)": [X_fe],
        "(2)": [X_fe, fe_bor],
        "(3)": [X_fe, fe_lender],
        "(4)": [X_fe, fe_leadarr],
        "(5)": [X_fe, fe_leadleft],
        "(6)": [X_fe, fe_bor, fe_lender],
        "(7)": [X_fe, fe_bor, fe_leadarr],
        "(8)": [X_fe, fe_bor, fe_leadleft],
    }

    col_data = {}
    for col_name, fe_parts in col_specs.items():
        print(f"\n{'=' * 60}\n  Model {col_name} (FE-only)\n{'=' * 60}")
        X_full = pd.concat(fe_parts, axis=1)
        X, y_clean = stabilize_fe(X_full, y)
        r2, adj_r2, n_obs, k = fe_only_fit(X, y_clean)

        cnt = {
            IY_LABEL: sum(c.startswith("iy_") for c in X.columns),
            BOR_LABEL: sum(c.startswith("b_")  for c in X.columns),
            LEN_LABEL: sum(c.startswith("l_")  for c in X.columns),
            LA_LABEL:  sum(c.startswith("la_") for c in X.columns),
            LL_LABEL:  sum(c.startswith("ll_") for c in X.columns),
        }
        fe_counts = {kk: vv for kk, vv in cnt.items() if vv > 0}

        values = [dv, f"{int(n_obs):,}", f"{r2:.4f}", f"{adj_r2:.4f}"]
        for label in fe_label_order:
            values.append(str(fe_counts[label]) if label in fe_counts else "")
        col_data[col_name] = values

        active = "  |  ".join(f"{kk.split(' FE')[0]}: {vv}" for kk, vv in fe_counts.items())
        print(f"  N = {int(n_obs):,}  |  R² = {r2:.4f}  |  Adj. R² = {adj_r2:.4f}  |  k = {k:,}")
        print(f"  {active}")

    index     = ["Dependent variable", "N", "R²", "Adj. R²"] + fe_label_order
    col_order = [f"({i})" for i in range(1, 9)]
    return pd.DataFrame(col_data, index=index)[col_order]


# ── Main ─────────────────────────────────────────────────────────────────────────

def run():
    print("Loading contracts …")
    df_full = pd.read_parquet(DATA_DIR / "fulldata.parquet")
    print(f"  {len(df_full):,} rows × {df_full.shape[1]} cols")

    lender_lists = load_lender_lists()
    lead_info    = load_lead_arranger_info()

    tables = {}
    for sheet_name, score_cols in DV_SPECS:
        tables[sheet_name] = build_table(df_full, lender_lists, lead_info, sheet_name, score_cols)

    OUT_DIR.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        for sheet_name, _ in DV_SPECS:
            tables[sheet_name].to_excel(xw, sheet_name=sheet_name)

    sheets = ", ".join(name for name, _ in DV_SPECS)
    print(f"\nSaved → {OUT_FILE}  (sheets: {sheets})")


if __name__ == "__main__":
    run()
