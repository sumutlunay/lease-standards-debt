"""
Step 3: Merge Dealscan, Compustat, and lender experience into the contracts dataset.

Sources
-------
Lender experience:   data/full_parent_event_selected_{12,24,36}.csv
Ratings:             https://ucdavis.box.com/shared/static/54165u2apn0i7uw841edzv66ez3e3gbj.csv  (external_variables.csv)
Base contracts:      data/ayung_upstream.parquet  — LLM-scored contracts
Dealscan:            data/dealscan.parquet         — lender×tranche level, collapsed here
Compustat:           data/compustat_2000_2025.parquet — firm×fiscal-year level
Output:              data/contracts.parquet

Merge order and join types
--------------------------
All contract-level joins use borrower_id + tranche_active_date as the composite key.

1. Lender experience anchor  — _12 file defines the sample (unique on borrower_id+date);
                               _24 and _36 left-joined on borrower_id + tranche_active_date.
                               Experience cols get _12/_24/_36 suffix.

2. Ratings (left join)       — current_rating, rating_date, rating_type from
                               full_event_selected_36_ratings.csv.

3. Base contracts (left join) — ayung_upstream.parquet; raw base has 4,501 duplicate
                               borrower_id+date rows (multiple LLM-scored exhibits per
                               tranche); deduplicated by keeping the row with the highest
                               sum of all seven claude_*_SCORE columns (highest contractual
                               intensity). Ties broken by: (1) claude_contract_type ==
                               'Original'; (2) highest sum of the five OBS-related scores
                               (SLB/SYN/OPL/VAR/RES). EDGAR link columns (text_link,
                               html_link, contract_file_link) are retained for audit.

4. Dealscan (left join)      — on borrower_id + tranche_active_date; collapsed from
                               lender×tranche to tranche level before merging.

5. Compustat (as-of left join) — on gvkey + tranche_active_date; attaches the most recent
                                  fiscal year end (datadate) <= tranche_active_date.
"""

from pathlib import Path
import re
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR.parent / "data"

CONTRACTS_SRC  = DATA_DIR / "ayung_upstream.parquet"
DEALSCAN_SRC   = DATA_DIR / "dealscan.parquet"
COMPUSTAT_SRC  = DATA_DIR / "compustat_2000_2025.parquet"
RATINGS_SRC    = "https://ucdavis.box.com/shared/static/54165u2apn0i7uw841edzv66ez3e3gbj.csv"
LENDER_EXP_SRC = {
    12: DATA_DIR / "full_parent_event_selected_12.csv",
    24: DATA_DIR / "full_parent_event_selected_24.csv",
    36: DATA_DIR / "full_parent_event_selected_36.csv",
}
OUT_FILE = DATA_DIR / "contracts.parquet"

# Composite key for all contract-level joins
MERGE_KEYS = ["borrower_id", "tranche_active_date"]

# Identifier columns in lender experience files that are not experience metrics
# (not renamed with window suffix)
LENDER_EXP_ID_COLS = {"borrower_id", "tranche_active_date", "cik", "gvkey", "file_link"}

# Constructed variables to bring from Dealscan (all built in 2_dealscan.py)
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

# Constructed variables to bring from Compustat (all built in 1_compustat.py)
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


# ── 1. Build lender experience anchor ────────────────────────────────────────
# Lender exp files are unique on borrower_id + tranche_active_date (no dedup needed).
# gvkey, cik, file_link from the _12 anchor are carried forward for downstream joins.
print("Loading lender experience files …")

anchor = None
for window, src_path in LENDER_EXP_SRC.items():
    le = pd.read_csv(src_path)
    le["tranche_active_date"] = pd.to_datetime(le["tranche_active_date"])

    # Rename experience columns with window suffix immediately
    exp_cols = [c for c in le.columns if c not in LENDER_EXP_ID_COLS]
    le = le.rename(columns={c: f"{c}_{window}" for c in exp_cols})

    if anchor is None:
        anchor = le
        print(f"  _12: {len(anchor):,} rows (unique on borrower_id + tranche_active_date)")
    else:
        renamed_exp = [f"{c}_{window}" for c in exp_cols]
        anchor = anchor.merge(le[MERGE_KEYS + renamed_exp], on=MERGE_KEYS, how="left")
        print(f"  _{window}: {len(anchor):,} rows after join")

n_base = len(anchor)
print(f"  Anchor: {n_base:,} rows × {anchor.shape[1]} cols")


# ── 2. Left join ratings ──────────────────────────────────────────────────────
print("\nLoading ratings …")
ratings = pd.read_csv(RATINGS_SRC, usecols=["borrower_id", "tranche_active_date",
                                              "current_rating", "rating_date", "rating_type"])
ratings["tranche_active_date"] = pd.to_datetime(ratings["tranche_active_date"])

anchor = anchor.merge(ratings, on=MERGE_KEYS, how="left")
n_rated = anchor["current_rating"].notna().sum()
print(f"  {n_rated:,} / {n_base:,} rows matched ({n_rated / n_base * 100:.1f}%)")

if len(anchor) != n_base:
    raise ValueError(f"Row count changed to {len(anchor):,} after ratings merge.")


# ── 3. Left join base contracts ───────────────────────────────────────────────
# Dedup strategy: keep the exhibit with the highest contractual intensity.
# Priority: (1) highest sum of all 7 claude_*_SCORE columns; (2) contract_type == 'Original';
# (3) highest sum of the 5 OBS-related scores (SLB/SYN/OPL/VAR/RES).
ALL_SCORE_COLS = [
    "claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
    "claude_VAR_SCORE", "claude_RES_SCORE",
    "claude_GAAP_OVERRIDE_SCORE", "claude_FREEZE_SCORE",
]
OBS_SCORE_COLS = ALL_SCORE_COLS[:5]  # first five: most directly tied to OBS measurement

print("\nLoading base contracts …")
contracts = pd.read_parquet(CONTRACTS_SRC)
contracts["tranche_active_date"] = pd.to_datetime(contracts["tranche_active_date"], errors="coerce")
print(f"  {len(contracts):,} rows × {contracts.shape[1]} cols (raw)")

contracts["_total_score"] = contracts[ALL_SCORE_COLS].fillna(0).sum(axis=1)
contracts["_obs_score"]   = contracts[OBS_SCORE_COLS].fillna(0).sum(axis=1)
contracts["_is_original"] = (contracts["claude_contract_type"].str.lower() == "original").astype(int)

contracts_sorted = contracts.sort_values(
    by=MERGE_KEYS + ["_total_score", "_is_original", "_obs_score"],
    ascending=[True, True, False, False, False],
)
contracts_dedup = contracts_sorted.drop_duplicates(subset=MERGE_KEYS, keep="first")
contracts_dedup = contracts_dedup.drop(columns=["_total_score", "_obs_score", "_is_original"])
print(f"  {len(contracts_dedup):,} rows after score-based dedup ({len(contracts) - len(contracts_dedup):,} duplicate rows dropped)")

# Drop columns already carried from the lender exp anchor to avoid collision.
# Rename contracts' file_link → contract_file_link so all three EDGAR link columns survive.
contracts_dedup = contracts_dedup.drop(columns=["gvkey", "cik"], errors="ignore")
if "file_link" in contracts_dedup.columns:
    contracts_dedup = contracts_dedup.rename(columns={"file_link": "contract_file_link"})

result = anchor.merge(contracts_dedup, on=MERGE_KEYS, how="left")
n_contracts_matched = result["cname"].notna().sum()

if len(result) != n_base:
    raise ValueError(f"Row count changed to {len(result):,} after base contracts merge.")

print(f"\n── Base contracts merge diagnostics ──────────────────────────────────")
print(f"  Anchor rows:             {n_base:>7,}")
print(f"  Matched to base:         {n_contracts_matched:>7,}  ({n_contracts_matched / n_base * 100:.1f}%)")
print(f"  Unmatched (left only):   {n_base - n_contracts_matched:>7,}  ({(n_base - n_contracts_matched) / n_base * 100:.1f}%)")


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


# ── 8. Recode credit ratings ──────────────────────────────────────────────────
# Long-term scale: D=1 … AAA=22; missing/NR/local-scale = 0
_LONG_TERM_SCALE = [
    'D', 'C', 'CC', 'CCC-', 'CCC', 'CCC+',
    'B-', 'B', 'B+', 'BB-', 'BB', 'BB+',
    'BBB-', 'BBB', 'BBB+', 'A-', 'A', 'A+',
    'AA-', 'AA', 'AA+', 'AAA',
]
_LONG_TERM_SET  = set(_LONG_TERM_SCALE)
_RATING_TO_NUM  = {r: i + 1 for i, r in enumerate(_LONG_TERM_SCALE)}

# Short-term → lowest approximate long-term equivalent
_SHORT_TERM_MAP = {
    'A-1+': 'AA-', 'A-1': 'A-', 'A-2': 'BBB-', 'A-3': 'BB+',
    'P-1': 'BBB+', 'P-1(Low)': 'BBB+', 'P-2': 'BBB-', 'P-3': 'BB+',
}

_NR_LABELS   = {'NR', 'NR/NR', 'NR prelim'}
_LOCAL_SCALE = {
    'B-1', 'B-2', 'B-3',
    'axBB', 'BB-pi', 'BBpi', 'Bpi', 'BBBpi',
    'ilAA+', 'ilA-1', 'ilA-', 'brAA+', 'kzA+', 'twAA-',
    'K-2', 'P-4(High)',
}

def _clean_rating(raw):
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if s in _NR_LABELS or s in _LOCAL_SCALE:
        return None
    # Remove structured-finance tag and "prelim" suffix (order matters: sf first)
    s = re.sub(r'\s*\(sf\)', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r'\s+prelim$', '', s, flags=re.IGNORECASE).strip()
    if s == 'SD':
        return 'D'
    # Dual ratings (long-term/short-term) — keep long-term portion
    if '/' in s:
        s = s.split('/')[0].strip()
    # Short-term ratings → lowest approximate long-term equivalent
    if s in _SHORT_TERM_MAP:
        return _SHORT_TERM_MAP[s]
    return s if s in _LONG_TERM_SET else None

print("\nRecoding credit ratings …")
result['modified_rating'] = result['current_rating'].apply(_clean_rating)
result['num_rating']      = result['modified_rating'].map(_RATING_TO_NUM).fillna(0).astype(int)
result['non_rated']       = (result['num_rating'] == 0).astype(int)

print(f"\n── Rating recode diagnostics ─────────────────────────────────────────")
print(f"  modified_rating non-missing: {result['modified_rating'].notna().sum():>7,}")
print(f"  num_rating distribution:")
print(result.groupby('num_rating')['modified_rating']
      .first().rename('modified_rating')
      .reset_index()
      .sort_values('num_rating')
      .to_string(index=False))
print(f"\n  non_rated == 1 (num_rating=0): {result['non_rated'].sum():>7,}")


# ── 9. Save ───────────────────────────────────────────────────────────────────
result.to_parquet(OUT_FILE, index=False)
print(f"\nSaved → {OUT_FILE}  ({len(result):,} rows × {result.shape[1]} cols)")
