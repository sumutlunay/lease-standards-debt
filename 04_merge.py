"""
Step 4: Merge Dealscan and Compustat into the contracts base dataset.

Sources
-------
Base contracts:  data/contracts_base.parquet  — output of 03_contracts.py
Dealscan:        data/dealscan.parquet         — lender×tranche level, collapsed here
Compustat:       data/compustat_2000_2025.parquet — firm×fiscal-year level
Output:          data/contracts.parquet

Merge order and join types
--------------------------
All joins use borrower_id + tranche_active_date or gvkey + tranche_active_date.

4. Dealscan (left join)      — on borrower_id + tranche_active_date; collapsed from
                               lender×tranche to tranche level before merging.

5. Compustat (as-of left join) — on gvkey + tranche_active_date; attaches the most recent
                                  fiscal year end (datadate) <= tranche_active_date.

6. Credit rating recode      — current_rating → modified_rating (cleaned string),
                               num_rating (1–22 numeric scale), non_rated indicator.
"""

from pathlib import Path
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR.parent / "data"

IN_FILE       = DATA_DIR / "contracts_base.parquet"
DEALSCAN_SRC  = DATA_DIR / "dealscan.parquet"
COMPUSTAT_SRC = DATA_DIR / "compustat_2000_2025.parquet"
OUT_FILE      = DATA_DIR / "contracts.parquet"

# Composite key for all contract-level joins
MERGE_KEYS = ["borrower_id", "tranche_active_date"]

# Constructed variables to bring from Dealscan (all built in 02_dealscan.py)
# lpc_deal_id retained for deal-level clustering in regressions
DEALSCAN_COLS = MERGE_KEYS + [
    "lpc_deal_id",
    "maturity",
    "log_lender_count",
    "log_interest",
    "log_deal_amount",
    "perf_pricing",
    "fin_covenant_count",
    "gen_covenant_count",
    "is_covenant_ratio",
    "secured",
]

# Constructed variables to bring from Compustat (all built in 01_compustat.py)
# sic retained for industry fixed effects
COMP_VARS = [
    "sic",
    "size",
    "profitability",
    "bsfixed",
    "liabilities",
    "offbslease",
    "logage",
    "btm",
    "capex",
    "loss",
    "rand",
    "divyield",
]


# ── Load base ─────────────────────────────────────────────────────────────────
print("Loading contracts base …")
result = pd.read_parquet(IN_FILE)
result["tranche_active_date"] = pd.to_datetime(result["tranche_active_date"])
n_base = len(result)
print(f"  {n_base:,} rows × {result.shape[1]} cols")


# ── 4. Load Dealscan and collapse to tranche level ────────────────────────────
print("\nLoading Dealscan …")
ds = pd.read_parquet(DEALSCAN_SRC)
ds["tranche_active_date"] = pd.to_datetime(ds["tranche_active_date"], errors="coerce")
print(f"  {len(ds):,} rows × {ds.shape[1]} cols (lender×tranche level)")

missing = [c for c in DEALSCAN_COLS if c not in ds.columns]
if missing:
    raise ValueError(f"Expected Dealscan columns not found: {missing}")

ds_cols = ds[DEALSCAN_COLS].copy()
ds_tranche = ds_cols.drop_duplicates(subset=MERGE_KEYS, keep="first")
print(f"  {len(ds_tranche):,} unique tranches after collapsing to tranche level")

n_dups = ds_tranche.duplicated(subset=MERGE_KEYS).sum()
if n_dups:
    raise ValueError(f"{n_dups:,} duplicate (borrower_id, tranche_active_date) remain in Dealscan after dedup")


# ── 5. Left join Dealscan ─────────────────────────────────────────────────────
print("Merging Dealscan …")
result = result.merge(ds_tranche, on=MERGE_KEYS, how="left")

if len(result) != n_base:
    raise ValueError(f"Row count changed to {len(result):,} after Dealscan merge.")

n_ds_matched = result["maturity"].notna().sum()
print(f"\n── Dealscan merge diagnostics ────────────────────────────────────────")
print(f"  Analysis sample:         {n_base:>7,}")
print(f"  Matched to Dealscan:     {n_ds_matched:>7,}  ({n_ds_matched / n_base * 100:.1f}%)")
print(f"  Unmatched (left only):   {n_base - n_ds_matched:>7,}  ({(n_base - n_ds_matched) / n_base * 100:.1f}%)")

print(f"\nDealscan variable coverage:")
for col in [c for c in DEALSCAN_COLS if c not in MERGE_KEYS]:
    n_nonmiss = result[col].notna().sum()
    print(f"  {col:<25} {n_nonmiss:>7,} non-missing  ({n_nonmiss / n_base * 100:.1f}%)")

# log_lender_count_alt: log of number_of_lenders from external_variables.csv (03_contracts.py).
# Kept alongside log_lender_count (Dealscan-derived) for comparison; the two diverge for ~6%
# of observations due to different source data and aggregation logic.
result["log_lender_count_alt"] = np.log(result["number_of_lenders"].replace(0, float("nan")))


# ── 6. Load and prepare Compustat ─────────────────────────────────────────────
print("\nLoading Compustat …")
comp = pd.read_parquet(COMPUSTAT_SRC)
comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")
comp["gvkey"] = comp["gvkey"].astype(str).str.strip()
print(f"  {len(comp):,} rows × {comp.shape[1]} cols (firm×fiscal-year level)")

comp = comp[["gvkey", "datadate", "fyear"] + COMP_VARS].copy()
comp = comp.sort_values("datadate").reset_index(drop=True)


# ── 7. As-of left join Compustat ──────────────────────────────────────────────
# gvkey from lender exp anchor is int64 (e.g. 2393); Compustat stores it zero-padded ("002393")
result["_row_order"] = range(len(result))
result["gvkey"] = result["gvkey"].astype(int).astype(str).str.zfill(6)

left_sorted = result.sort_values("tranche_active_date").reset_index(drop=True)

result = pd.merge_asof(
    left_sorted,
    comp.rename(columns={"datadate": "comp_datadate"}),
    left_on="tranche_active_date",
    right_on="comp_datadate",
    by="gvkey",
    direction="backward",
)

result = (
    result
    .sort_values("_row_order")
    .drop(columns="_row_order")
    .reset_index(drop=True)
)

if len(result) != n_base:
    raise ValueError(f"Row count changed to {len(result):,} after Compustat merge.")

n_comp_matched = result["size"].notna().sum()
print(f"\n── Compustat merge diagnostics ───────────────────────────────────────")
print(f"  Analysis sample:         {n_base:>7,}")
print(f"  Matched to Compustat:    {n_comp_matched:>7,}  ({n_comp_matched / n_base * 100:.1f}%)")
print(f"  Unmatched (left only):   {n_base - n_comp_matched:>7,}  ({(n_base - n_comp_matched) / n_base * 100:.1f}%)")

print(f"\nCompustat variable coverage:")
for col in COMP_VARS:
    n_nonmiss = result[col].notna().sum()
    print(f"  {col:<25} {n_nonmiss:>7,} non-missing  ({n_nonmiss / n_base * 100:.1f}%)")
print(f"  {'sic (unique values)':<25} {result['sic'].nunique():>7,} unique SIC codes")


# ── Save ──────────────────────────────────────────────────────────────────────
result.to_parquet(OUT_FILE, index=False)
print(f"\nSaved → {OUT_FILE}  ({len(result):,} rows × {result.shape[1]} cols)")
