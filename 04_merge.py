"""
Step 4: Merge Dealscan and Compustat into the contracts base dataset.

Sources
-------
Base contracts:  data/contracts_base.parquet  — output of 03_contracts.py
Dealscan:        data/dealscan.parquet         — lender×tranche level, collapsed here
Compustat:       data/compustat_1998_2025.parquet — firm×fiscal-year level
Output:          data/contracts.parquet

Merge order and join types
--------------------------
All joins use borrower_id + tranche_active_date or gvkey + tranche_active_date.

4. Dealscan (left join) — Dealscan is lender×tranche level, and one
   borrower_id + tranche_active_date can span several tranches (and, rarely, several
   deals signed the same day). It is collapsed to one row per contract in two steps:
     (a) Pin the deal. Keep only tranches whose lpc_deal_id equals the contract's
         lpc_deal_id_extvars — the deal id carried from the external-variables file in
         03_contracts.py — which resolves the rare same-day multi-deal case exactly.
     (b) Pick the tranche. Within the pinned deal, keep the tranche with the largest
         tranche_amount (ties broken by lpc_tranche_id for reproducibility).
   The join therefore matches on borrower_id + tranche_active_date + deal id.

5. Compustat (windowed as-of left join) — on gvkey. To avoid look-ahead bias, a loan is
                                  matched to the most recent fiscal year end (datadate)
                                  that sits in the window
                                    tranche_active_date - 15 months <= datadate
                                                                    <= tranche_active_date - 3 months.
                                  The 3-month upper gap approximates the 10-K filing lag
                                  (financials are not public until ~90 days after FYE); the
                                  15-month lower bound caps staleness.

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
COMPUSTAT_SRC = DATA_DIR / "compustat_1998_2025.parquet"
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

# Deal id carried from the external-variables file (03_contracts.py). Used to pin which
# Dealscan deal a contract belongs to when a borrower signs more than one deal the same day.
DEAL_KEY_LEFT = "lpc_deal_id_extvars"

# Extra Dealscan columns needed only to choose one tranche per deal; dropped after collapse.
COLLAPSE_COLS = ["lpc_tranche_id", "tranche_amount"]

# Constructed variables to bring from Compustat (all built in 01_compustat.py)
# sic retained for industry fixed effects
# at (raw total assets, $ millions) retained to scale cumulative_bond_proceeds below
COMP_VARS = [
    "sic",
    "at",
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

missing = [c for c in DEALSCAN_COLS + COLLAPSE_COLS if c not in ds.columns]
if missing:
    raise ValueError(f"Expected Dealscan columns not found: {missing}")
if DEAL_KEY_LEFT not in result.columns:
    raise ValueError(f"'{DEAL_KEY_LEFT}' missing from contracts_base — needed to pin the deal.")

# Collapse lender×tranche → one row per (borrower_id, tranche_active_date, lpc_deal_id):
# within each deal keep the largest-amount tranche (ties broken by lpc_tranche_id). Every
# column in DEALSCAN_COLS is deal/tranche-level and constant across a tranche's lender rows,
# so keeping one lender row of the chosen tranche is exact.
DEAL_KEYS = MERGE_KEYS + ["lpc_deal_id"]
ds_cols = ds[DEALSCAN_COLS + COLLAPSE_COLS].copy()
ds_deal = (
    ds_cols.sort_values(
        ["tranche_amount", "lpc_tranche_id"], ascending=[False, True], na_position="last"
    )
    .drop_duplicates(subset=DEAL_KEYS, keep="first")
    .drop(columns=COLLAPSE_COLS)
)
print(f"  {len(ds_deal):,} (borrower, date, deal) rows after largest-tranche pick")

n_dups = ds_deal.duplicated(subset=DEAL_KEYS).sum()
if n_dups:
    raise ValueError(f"{n_dups:,} duplicate {tuple(DEAL_KEYS)} remain in Dealscan after collapse")


# ── 5. Left join Dealscan (pinned to the contract's deal) ─────────────────────
# Match on borrower_id + tranche_active_date + deal id, taking the contract's deal id from
# lpc_deal_id_extvars (03) and Dealscan's from lpc_deal_id via a temporary join column.
# Many-to-one, so the row count is preserved; for every matched row lpc_deal_id (Dealscan)
# == lpc_deal_id_extvars afterwards.
print("Merging Dealscan …")
result["_deal_pin"] = result[DEAL_KEY_LEFT]
ds_deal["_deal_pin"] = ds_deal["lpc_deal_id"]
result = result.merge(ds_deal, on=MERGE_KEYS + ["_deal_pin"], how="left").drop(columns="_deal_pin")

if len(result) != n_base:
    raise ValueError(f"Row count changed to {len(result):,} after Dealscan merge.")

# Consistency guard: the merged deal id must equal the pin for every matched row.
_matched = result["lpc_deal_id"].notna()
_mismatch = int((result.loc[_matched, "lpc_deal_id"] != result.loc[_matched, DEAL_KEY_LEFT]).sum())
if _mismatch:
    raise ValueError(f"{_mismatch:,} matched rows where lpc_deal_id != {DEAL_KEY_LEFT}")

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


# ── 7. Windowed as-of left join Compustat (A2: no look-ahead) ─────────────────
# Match each loan to the most recent fiscal year end (datadate) that was already public when
# the deal was signed, i.e. in the window
#     tranche_active_date - 15 months  <=  datadate  <=  tranche_active_date - 3 months.
# Implemented as a backward as-of on the 3-month cutoff (→ most recent datadate <= loan-3mo),
# then dropping any match older than the 15-month bound. DateOffset keeps the bounds on exact
# calendar months rather than day approximations.
REPORTING_LAG = pd.DateOffset(months=3)    # earliest gap between FYE and loan (10-K filing lag)
STALENESS_CAP = pd.DateOffset(months=15)   # oldest FYE still usable

# gvkey from lender exp anchor is int64 (e.g. 2393); Compustat stores it zero-padded ("002393")
result["_row_order"] = range(len(result))
result["gvkey"] = result["gvkey"].astype(int).astype(str).str.zfill(6)
result["_cutoff_hi"] = result["tranche_active_date"] - REPORTING_LAG   # latest allowed datadate
result["_cutoff_lo"] = result["tranche_active_date"] - STALENESS_CAP   # earliest allowed datadate

# Backward as-of on the 3-month cutoff → most recent datadate <= (loan_date - 3 months)
left_sorted = result.sort_values("_cutoff_hi").reset_index(drop=True)

result = pd.merge_asof(
    left_sorted,
    comp.rename(columns={"datadate": "comp_datadate"}),
    left_on="_cutoff_hi",
    right_on="comp_datadate",
    by="gvkey",
    direction="backward",
)

# Enforce the 15-month lower bound: null out any Compustat match older than (loan_date - 15 months)
too_stale = result["comp_datadate"].notna() & (result["comp_datadate"] < result["_cutoff_lo"])
n_stale_dropped = int(too_stale.sum())
result.loc[too_stale, "comp_datadate"] = pd.NaT
result.loc[too_stale, ["fyear"] + COMP_VARS] = np.nan

result = (
    result
    .sort_values("_row_order")
    .drop(columns=["_row_order", "_cutoff_hi", "_cutoff_lo"])
    .reset_index(drop=True)
)

if len(result) != n_base:
    raise ValueError(f"Row count changed to {len(result):,} after Compustat merge.")

n_comp_matched = result["size"].notna().sum()
print(f"\n── Compustat merge diagnostics ───────────────────────────────────────")
print(f"  Analysis sample:         {n_base:>7,}")
print(f"  Matched to Compustat:    {n_comp_matched:>7,}  ({n_comp_matched / n_base * 100:.1f}%)")
print(f"  Unmatched (left only):   {n_base - n_comp_matched:>7,}  ({(n_base - n_comp_matched) / n_base * 100:.1f}%)")
print(f"  Dropped by 15-mo cap:    {n_stale_dropped:>7,}  (match existed <= loan-3mo but older than loan-15mo)")

print(f"\nCompustat variable coverage:")
for col in COMP_VARS:
    n_nonmiss = result[col].notna().sum()
    print(f"  {col:<25} {n_nonmiss:>7,} non-missing  ({n_nonmiss / n_base * 100:.1f}%)")
print(f"  {'sic (unique values)':<25} {result['sic'].nunique():>7,} unique SIC codes")


# ── 7a. Scale cumulative bond proceeds by total assets ────────────────────────
# bond_proceeds_scaled = cumulative_bond_proceeds / total assets, as a unitless ratio.
# UNIT RECONCILIATION: cumulative_bond_proceeds is in $ THOUSANDS (from FISD, via 03);
# Compustat `at` is in $ MILLIONS. Convert `at` to thousands (× 1000) before dividing,
# else the ratio is off by 1,000×. Divide only where at > 0 (matches how 01 builds its
# other asset-scaled ratios); at missing/≤0 → NaN (cannot scale, drops from those tests).
_at_positive = result["at"].where(result["at"] > 0)
result["bond_proceeds_scaled"] = result["cumulative_bond_proceeds"] / (_at_positive * 1000)
_n_scaled = int(result["bond_proceeds_scaled"].notna().sum())
_n_pos    = int((result["bond_proceeds_scaled"] > 0).sum())
print(f"\n── Bond proceeds scaling ─────────────────────────────────────────────")
print(f"  bond_proceeds_scaled = cumulative_bond_proceeds($000s) / (at($M) × 1000)")
print(f"  non-missing (at > 0):    {_n_scaled:>7,}  ({_n_scaled / n_base * 100:.1f}%)")
print(f"  strictly positive:       {_n_pos:>7,}  ({_n_pos / n_base * 100:.1f}%)")
print(f"  median | p99 | max:      "
      f"{result['bond_proceeds_scaled'].median():.4f} | "
      f"{result['bond_proceeds_scaled'].quantile(0.99):.4f} | "
      f"{result['bond_proceeds_scaled'].max():.4f}")


# ── Save ──────────────────────────────────────────────────────────────────────
result.to_parquet(OUT_FILE, index=False)
print(f"\nSaved → {OUT_FILE}  ({len(result):,} rows × {result.shape[1]} cols)")
