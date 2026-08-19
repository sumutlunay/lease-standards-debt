"""
table2_fe_decomposition.py  —  Table 2: FE-only variance decomposition
=======================================================================

How much of each off-balance-sheet lease-covenant dummy is explained by the fixed-effect
structure ALONE, with no regressors on the right-hand side. Eight FE structures × five
dependent variables (one sheet each); every column is a pure projection of the DV onto its
fixed effects, so each reports only N, R², Adj. R² and the FE counts.

    (1) Industry×Year                     (5) Industry×Year + Lead-left
    (2) Industry×Year + Borrower          (6) Industry×Year + Borrower + Lender
    (3) Industry×Year + Lender            (7) Industry×Year + Borrower + Lead-arranger
    (4) Industry×Year + Lead-arranger     (8) Industry×Year + Borrower + Lead-left

Lender / Lead-arranger FEs are multi-hot (a contract has many); Lead-left is a single
category per deal.

SAMPLE — the FULL scored-contract sample (14,584), not the 11,184 estimation sample the
regression tables use. With no regressors there is no completeness filter to apply, and FE
dummies are always finite so no rows drop. ⚠ R² here is therefore NOT comparable to the
regression tables' R².

R² is CENTERED (1 − SSR/SST about the mean); Adj. R² penalizes by the identified FE rank,
after the same singleton/duplicate/linear-dependence guard the regression scripts use. Fit
with no separate intercept — the complete Industry×Year block already spans the constant.

Rehash of exploratory/06b_rq1_table2.py, trimmed to the table (console output and docstring
condensed). Output is byte-identical to that script's.

Input : data/contracts.parquet, data/dealscan_raw.parquet
Output: output/tables/table2.xlsx  (sheets: SLB, SYN, OPL, VAR-RES, ALL)
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent   # code/ — this script lives in code/manuscript/
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
_SHARE_RE  = re.compile(r"\s*\d{1,3}(?:\.\d+)?\s*%")


# ── DealScan lead-arranger / lead-left parsing ───────────────────────────────────

def parse_arrangers(s) -> list:
    if pd.isna(s) or str(s).strip() in ("", "None"):
        return []
    seen, out = set(), []
    for tok in _SHARE_RE.sub("", str(s)).split(","):
        t = tok.strip()
        if t and t != "None" and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def clean_one_name(s):
    if pd.isna(s) or str(s).strip() in ("", "None"):
        return None
    return _SHARE_RE.sub("", str(s)).split(",")[0].strip()


# ── Fixed-effect builders ────────────────────────────────────────────────────────

def make_industry_year_fe(df: pd.DataFrame, sic_digits: int = 2):
    sic_str  = df["sic"].fillna(0).astype(int).astype(str).str.zfill(4).str[:sic_digits]
    ind_year = sic_str + "_" + df["contract_year"].astype(str)
    return pd.get_dummies(ind_year, prefix="iy", dtype=float), int(ind_year.nunique())


def make_borrower_fe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df["borrower_id"].astype("Int64"), prefix="b", dtype=float)


def make_lender_multi_hot(df: pd.DataFrame, lender_lists: pd.Series) -> pd.DataFrame:
    s = lender_lists.explode().dropna()
    if s.empty:
        return pd.DataFrame(index=df.index)
    oh = (pd.get_dummies(s.astype(int), prefix="l", prefix_sep="_", dtype=float)
          .groupby(level=0).max())
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


# ── Shared DealScan inputs ───────────────────────────────────────────────────────

def load_lender_lists() -> pd.Series:
    ds = pd.read_parquet(DATA_DIR / "dealscan_raw.parquet",
                         columns=MERGE_KEYS + ["lender_parent_id"])
    ds["tranche_active_date"] = pd.to_datetime(ds["tranche_active_date"], errors="coerce")
    ds = ds.dropna(subset=["lender_parent_id"])
    ds["lender_parent_id"] = ds["lender_parent_id"].astype(int)
    return ds.groupby(MERGE_KEYS)["lender_parent_id"].apply(list).rename("lender_ids")


def load_lead_arranger_info() -> pd.DataFrame:
    ds = pd.read_parquet(DATA_DIR / "dealscan_raw.parquet",
                         columns=MERGE_KEYS + ["lead_arranger", "lead_left"])
    ds["tranche_active_date"] = pd.to_datetime(ds["tranche_active_date"], errors="coerce")
    ds = ds.dropna(subset=MERGE_KEYS)
    for c in ["lead_arranger", "lead_left"]:
        ds[c] = ds[c].where(ds[c].astype(str).str.strip().ne("None"))
    g = ds.groupby(MERGE_KEYS)[["lead_arranger", "lead_left"]].first()

    g["arr_names"] = g["lead_arranger"].apply(parse_arrangers)
    g["ll_native"] = g["lead_left"].apply(clean_one_name)
    # Fall back to the sole arranger when lead_left is absent but unambiguous.
    g["lead_left_name"] = g.apply(
        lambda r: r["ll_native"] if r["ll_native"]
        else (r["arr_names"][0] if len(r["arr_names"]) == 1 else "__NONE__"), axis=1)
    print(f"  lead-arranger/left: {len(g):,} deal-keys; lead-left assigned on "
          f"{(g['lead_left_name'] != '__NONE__').mean() * 100:.1f}%")
    return g[["arr_names", "lead_left_name"]]


# ── Design cleaner + FE-only fit ─────────────────────────────────────────────────

def drop_dependent_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are exact linear combinations of columns to their LEFT (unpivoted
    QR), so the identified FE rank is well-defined. Estimate-preserving (span unchanged)."""
    Xv   = X.to_numpy(dtype=float)
    _, R = np.linalg.qr(Xv, mode="reduced")
    diag = np.abs(np.diag(R))
    keep = diag > diag.max() * max(Xv.shape) * np.finfo(float).eps
    if (n_drop := int((~keep).sum())):
        print(f"      rank guard: dropped {n_drop} dependent column(s) "
              f"({X.shape[1]} → {int(keep.sum())})")
    return X.loc[:, keep]


def stabilize_fe(X: pd.DataFrame, y: pd.Series):
    """Drop non-finite rows, then constant / duplicate / singleton / dependent FE columns."""
    row_ok = (np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
              & np.isfinite(y.to_numpy(dtype=float)))
    if (n_dropped := int((~row_ok).sum())):
        print(f"      stabilize: {n_dropped} non-finite row(s) dropped")
    X, y = X.loc[row_ok], y.loc[row_ok]

    X = X.loc[:, X.std(ddof=0) > 0]      # constant
    X = X.loc[:, ~X.T.duplicated()]      # duplicate
    X = X.loc[:, X.sum(axis=0) != 1]     # singleton
    return drop_dependent_columns(X).astype(float), y


def fe_only_fit(X: pd.DataFrame, y: pd.Series):
    """Project y onto the full-rank FE design → (centered R², adj R², n, k)."""
    Xv, yv = X.to_numpy(dtype=float), y.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid  = yv - Xv @ beta
    yc     = yv - yv.mean()
    r2     = 1.0 - float(resid @ resid) / max(float(yc @ yc), 1e-300)
    n, k   = Xv.shape
    adj    = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else float("nan")
    return r2, adj, n, k


# ── Per-DV table builder ─────────────────────────────────────────────────────────

def build_table(df_full, lender_lists, lead_info, sheet_name, score_cols) -> pd.DataFrame:
    dv = f"{sheet_name}_score_dummy"

    df = df_full[df_full["claude_is_debt_contract"] == "Y"]
    df = df[df[score_cols].notna().all(axis=1)].copy()
    df[dv]              = (df[score_cols] > 0).any(axis=1).astype(float)
    df["contract_year"] = df["tranche_active_date"].dt.year

    # Industry×Year, backing off SIC granularity if the cells would exhaust the sample.
    N, sic_digits = len(df), 2
    fe_iy, n_cells = make_industry_year_fe(df, sic_digits)
    while n_cells >= N - 5 and sic_digits > 1:
        sic_digits -= 1
        fe_iy, n_cells = make_industry_year_fe(df, sic_digits)
    X_fe = fe_iy.loc[:, (fe_iy != 0).any(axis=0)]

    fe_bor = make_borrower_fe(df)

    df = df.join(lender_lists, on=MERGE_KEYS)
    df["lender_ids"] = df["lender_ids"].apply(lambda x: x if isinstance(x, list) else [])
    fe_lender = make_lender_multi_hot(df, df["lender_ids"])

    df = df.join(lead_info, on=MERGE_KEYS)
    df["arr_names"]      = df["arr_names"].apply(lambda x: x if isinstance(x, list) else [])
    df["lead_left_name"] = df["lead_left_name"].where(df["lead_left_name"].notna(), "__NONE__")
    fe_leadarr  = make_name_multi_hot(df["arr_names"], prefix="la_")
    fe_leadleft = make_single_cat_fe(df["lead_left_name"], prefix="ll_")

    # (row label, column-name prefix) — drives both the FE-count rows and their order.
    FE_BLOCKS = [
        (f"Industry×Year FEs (SIC {sic_digits}-digit)", "iy_"),
        ("Borrower FEs",                                "b_"),
        ("Lender FEs",                                  "l_"),
        ("Lead-arranger FEs",                           "la_"),
        ("Lead-left FEs",                               "ll_"),
    ]
    col_specs = {
        "(1)": [X_fe],                        "(5)": [X_fe, fe_leadleft],
        "(2)": [X_fe, fe_bor],                "(6)": [X_fe, fe_bor, fe_lender],
        "(3)": [X_fe, fe_lender],             "(7)": [X_fe, fe_bor, fe_leadarr],
        "(4)": [X_fe, fe_leadarr],            "(8)": [X_fe, fe_bor, fe_leadleft],
    }

    print(f"\n{sheet_name} ({dv}): N={N:,} | IY={X_fe.shape[1]} Bor={fe_bor.shape[1]} "
          f"Len={fe_lender.shape[1]} LeadArr={fe_leadarr.shape[1]} LeadLeft={fe_leadleft.shape[1]}")

    col_data = {}
    for col_name in [f"({i})" for i in range(1, 9)]:
        X, y_clean = stabilize_fe(pd.concat(col_specs[col_name], axis=1), df[dv])
        r2, adj_r2, n_obs, k = fe_only_fit(X, y_clean)
        counts = {lab: sum(c.startswith(pre) for c in X.columns) for lab, pre in FE_BLOCKS}
        col_data[col_name] = ([dv, f"{int(n_obs):,}", f"{r2:.4f}", f"{adj_r2:.4f}"]
                              + [str(counts[lab]) if counts[lab] else "" for lab, _ in FE_BLOCKS])
        active = "  ".join(f"{lab.split(' FE')[0]}={counts[lab]}" for lab, _ in FE_BLOCKS
                           if counts[lab])
        print(f"  {col_name}  N={int(n_obs):,}  R²={r2:.4f}  Adj.R²={adj_r2:.4f}  k={k:,}  | {active}")

    index = ["Dependent variable", "N", "R²", "Adj. R²"] + [lab for lab, _ in FE_BLOCKS]
    return pd.DataFrame(col_data, index=index)[[f"({i})" for i in range(1, 9)]]


def run() -> None:
    print("Loading contracts …")
    df_full = pd.read_parquet(DATA_DIR / "contracts.parquet")
    print(f"  {len(df_full):,} rows × {df_full.shape[1]} cols")
    lender_lists = load_lender_lists()
    lead_info    = load_lead_arranger_info()

    tables = {name: build_table(df_full, lender_lists, lead_info, name, cols)
              for name, cols in DV_SPECS}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as xw:
        for name, _ in DV_SPECS:
            tables[name].to_excel(xw, sheet_name=name)
    print(f"\nSaved → {OUT_FILE}  (sheets: {', '.join(n for n, _ in DV_SPECS)})")


if __name__ == "__main__":
    run()
