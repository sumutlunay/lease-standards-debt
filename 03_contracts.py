"""
Step 3: Merge lender experience, ratings, and LLM-scored contracts.

Sources
-------
Lender experience:   Box (full_parent_event_selected_{12,24,36}.csv) — fetched at runtime
Ratings/controls:    https://ucdavis.box.com/shared/static/54165u2apn0i7uw841edzv66ez3e3gbj.csv  (external_variables.csv)
Base contracts:      data/contracts_dv.parquet  — LLM-scored contracts
ASC adoption:        https://ucdavis.box.com/shared/static/2xb866l3q86x85hc77huzbd2spngdufi.csv  (asc_adoption_counts.csv)
Output:              data/contracts_base.parquet

Merge order and join types
--------------------------
All contract-level joins use borrower_id + tranche_active_date as the composite key,
except the ASC adoption file which joins on cik (firm level).

1. Lender experience anchor  — _12 file defines the sample (unique on borrower_id+date);
                               _24 and _36 left-joined on borrower_id + tranche_active_date.
                               Experience cols get _12/_24/_36 suffix.

2. Ratings/controls (left join) — from external_variables.csv; pulls all non-identifier
                                   columns: ratings, relationship vars, deal metadata,
                                   plus lpc_deal_id (renamed lpc_deal_id_extvars to avoid
                                   collision with Dealscan's own lpc_deal_id in 04_merge.py).

3. Base contracts (left join) — contracts_dv.parquet; raw base has duplicate
                               borrower_id+date rows (multiple LLM-scored exhibits per
                               tranche); deduplicated by keeping the row with the highest
                               sum of all seven claude_*_SCORE columns (highest contractual
                               intensity). Ties broken by: (1) claude_contract_type ==
                               'Original'; (2) highest sum of the five OBS-related scores
                               (SLB/SYN/OPL/VAR/RES). EDGAR link columns (text_link,
                               html_link, contract_file_link) are retained for audit.

4. ASC adoption (left join)  — asc_adoption_counts.csv; already unique on cik
                               (4,185 obs). Joins on cik. Adds adoption_date,
                               date_source, description, and pre/post contract and
                               amendment counts.
"""

from pathlib import Path
import re
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR.parent / "data"

CONTRACTS_SRC  = DATA_DIR / "contracts_dv.parquet"
RATINGS_SRC    = "https://ucdavis.box.com/shared/static/54165u2apn0i7uw841edzv66ez3e3gbj.csv"
ASC_SRC        = "https://ucdavis.box.com/shared/static/2xb866l3q86x85hc77huzbd2spngdufi.csv"
LENDER_EXP_SRC = {
    12: "https://ucdavis.box.com/shared/static/njlfl5ehm0yjk297esxlt9l2og4b4nks.csv",
    24: "https://ucdavis.box.com/shared/static/huuyiwt00rc7hxmjmxhf7a70p87mozu8.csv",
    36: "https://ucdavis.box.com/shared/static/vd8wfkmi8zci5p332pqn1n5ugxusp7aa.csv",
}
OUT_FILE           = DATA_DIR / "contracts_base.parquet"
VAR_LIST_FILE      = REPO_DIR.parent / "documentation" / "variables_claude_output.txt"
CONTRACTS_VAR_FILE = REPO_DIR.parent / "documentation" / "variables_contracts.txt"

# Composite key for all contract-level joins
MERGE_KEYS = ["borrower_id", "tranche_active_date"]

# Identifier columns in lender experience files that are not experience metrics
# (not renamed with window suffix)
LENDER_EXP_ID_COLS = {"borrower_id", "tranche_active_date", "cik", "gvkey", "file_link", "lpc_deal_id"}

# Constructed variables to bring from base contracts scoring
ALL_SCORE_COLS = [
    "claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
    "claude_VAR_SCORE", "claude_RES_SCORE",
    "claude_GAAP_OVERRIDE_SCORE", "claude_FREEZE_SCORE",
]
OBS_SCORE_COLS = ALL_SCORE_COLS[:5]  # first five: most directly tied to OBS measurement


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
        anchor = le.drop(columns=["lpc_deal_id"], errors="ignore")
        print(f"  _12: {len(anchor):,} rows (unique on borrower_id + tranche_active_date)")
    else:
        renamed_exp = [f"{c}_{window}" for c in exp_cols]
        anchor = anchor.merge(le[MERGE_KEYS + renamed_exp], on=MERGE_KEYS, how="left")
        print(f"  _{window}: {len(anchor):,} rows after join")

n_base = len(anchor)
print(f"  Anchor: {n_base:,} rows × {anchor.shape[1]} cols")


# ── 2. Left join ratings and external controls ────────────────────────────────
print("\nLoading ratings and external controls …")
ratings = pd.read_csv(RATINGS_SRC, usecols=[
    "borrower_id", "tranche_active_date",
    "lpc_deal_id",
    "past_relationship_count", "number_of_lenders", "relationship_freq",
    "current_rating", "rating_date", "rating_type",
    "deal_active_date", "amendment_seq",
])
ratings["tranche_active_date"] = pd.to_datetime(ratings["tranche_active_date"])
# Renamed to avoid collision with Dealscan's own lpc_deal_id, added later in 04_merge.py
ratings = ratings.rename(columns={"lpc_deal_id": "lpc_deal_id_extvars"})

anchor = anchor.merge(ratings, on=MERGE_KEYS, how="left")
n_rated = anchor["current_rating"].notna().sum()
print(f"  {n_rated:,} / {n_base:,} rows matched ({n_rated / n_base * 100:.1f}%)")

if len(anchor) != n_base:
    raise ValueError(f"Row count changed to {len(anchor):,} after ratings merge.")


# ── 2b. Recode credit ratings ─────────────────────────────────────────────────
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
# Two parallel rating constructions are produced:
#   • UNRESTRICTED (primary, used in the analyses): the borrower's latest S&P
#     rating regardless of how old it is — modified_rating / num_rating / non_rated.
#   • 36-MONTH RESTRICTED (robustness): ratings older than 36 calendar months
#     before the tranche active date are treated as not-rated —
#     modified_rating_36m / num_rating_36m / non_rated_36m.
# rating_stale flags which observations differ between the two constructions.

# ── Unrestricted (primary) ────────────────────────────────────────────────────
anchor['modified_rating'] = anchor['current_rating'].apply(_clean_rating)
anchor['num_rating']      = anchor['modified_rating'].map(_RATING_TO_NUM).fillna(0).astype(int)
anchor['non_rated']       = (anchor['num_rating'] == 0).astype(int)

# ── 36-month staleness flag ───────────────────────────────────────────────────
# If tranche_active_date falls more than 36 calendar months after rating_date,
# the rating is too stale to reflect credit quality at contracting time.
anchor['rating_date'] = pd.to_datetime(anchor['rating_date'], errors='coerce')
_rating_cutoff = anchor['rating_date'] + pd.DateOffset(months=36)
anchor['rating_stale'] = (
    anchor['rating_date'].notna() & (anchor['tranche_active_date'] > _rating_cutoff)
)
n_stale = anchor['rating_stale'].sum()

# ── 36-month restricted (robustness) ──────────────────────────────────────────
# Copy of the primary construction with stale ratings nulled out / set not-rated.
anchor['modified_rating_36m'] = anchor['modified_rating'].where(~anchor['rating_stale'], None)
anchor['num_rating_36m']      = anchor['num_rating'].where(~anchor['rating_stale'], 0).astype(int)
anchor['non_rated_36m']       = (anchor['num_rating_36m'] == 0).astype(int)

print(f"\n── Rating recode diagnostics ─────────────────────────────────────────")
print(f"  [primary/unrestricted]")
print(f"    modified_rating non-missing: {anchor['modified_rating'].notna().sum():>7,}")
print(f"    non_rated == 1 (num_rating=0): {anchor['non_rated'].sum():>7,}")
print(f"  [36-month restricted]")
print(f"    ratings flagged stale (>36mo before tranche_active_date): {n_stale:>7,}")
print(f"    modified_rating_36m non-missing: {anchor['modified_rating_36m'].notna().sum():>7,}")
print(f"    non_rated_36m == 1: {anchor['non_rated_36m'].sum():>7,}")
print(f"  num_rating (unrestricted) distribution:")
print(anchor.groupby('num_rating')['modified_rating']
      .first().rename('modified_rating')
      .reset_index()
      .sort_values('num_rating')
      .to_string(index=False))


# ── 3. Left join base contracts ───────────────────────────────────────────────
# Dedup strategy: keep the exhibit with the highest contractual intensity.
# Priority: (1) highest sum of all 7 claude_*_SCORE columns; (2) contract_type == 'Original';
# (3) highest sum of the 5 OBS-related scores (SLB/SYN/OPL/VAR/RES).
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
contracts_dedup = contracts_dedup.drop(columns=["gvkey", "cik", "gvkey_original", "cik_original"], errors="ignore")
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


# ── 4. Left join ASC adoption counts ─────────────────────────────────────────
# Firm-level file — already unique on cik (4,185 obs, 4,185 unique CIKs).
print("\nLoading ASC adoption counts …")
asc = pd.read_csv(ASC_SRC, usecols=[
    "cik",
    "adoption_date", "date_source", "description",
    "pre_contract_count", "post_contract_count",
    "pre_amendment_count", "post_amendment_count",
])
asc["cik"] = asc["cik"].astype(str).str.strip()

result["cik"] = result["cik"].astype(str).str.strip()
result = result.merge(asc, on="cik", how="left")

if len(result) != n_base:
    raise ValueError(f"Row count changed to {len(result):,} after ASC adoption merge.")

n_asc_matched = result["adoption_date"].notna().sum()
print(f"\n── ASC adoption merge diagnostics ────────────────────────────────────")
print(f"  Anchor rows:             {n_base:>7,}")
print(f"  Matched to ASC file:     {n_asc_matched:>7,}  ({n_asc_matched / n_base * 100:.1f}%)")
print(f"  Unmatched (left only):   {n_base - n_asc_matched:>7,}  ({(n_base - n_asc_matched) / n_base * 100:.1f}%)")


# ── Save ──────────────────────────────────────────────────────────────────────
result.to_parquet(OUT_FILE, index=False)
print(f"\nSaved → {OUT_FILE}  ({len(result):,} rows × {result.shape[1]} cols)")


# ── Write variable list ───────────────────────────────────────────────────────
def write_variable_list(df: pd.DataFrame) -> None:
    # Human-readable descriptions for columns sourced from contracts_dv.parquet
    descriptions = {
        "borrower_id":                 "Dealscan borrower ID (merge key)",
        "tranche_active_date":         "Dealscan tranche date (merge key)",
        "contract_id":                 "Internal sequential counter (1–29); NOT a Dealscan ID",
        "cname":                       "Company name",
        "ftype":                       "Filing type (8-K 56%, 10-Q 27%, 10-K 14%)",
        "fdate":                       "Filing date",
        "text_link":                   "Link to plain-text filing",
        "html_link":                   "Link to HTML filing",
        "contract_file_link":          "Link to filing index (renamed from file_link to avoid collision with lender exp anchor)",
        "line1":                       "Evidence line 1 from filing",
        "line2":                       "Evidence line 2 from filing",
        "line3":                       "Evidence line 3 from filing",
        "line4":                       "Evidence line 4 from filing",
        "line5":                       "Evidence line 5 from filing",
        "first_date":                  "First date extracted from filing",
        "newest_date_first20":         "Newest date in first 20 lines of filing",
        "claude_timestamp_utc":        "When the LLM call was made (UTC)",
        "claude_finalized_by":         "Model/version that finalized the record",
        "claude_validation_ok":        "Whether output passed validation",
        "claude_validation_errors":    "Validation error messages, if any",
        "claude_evidence_pack_chars":  "Character count of evidence sent to LLM",
        "claude_evidence_pack_hash":   "Hash of evidence pack (dedup check)",
        "claude_request_id":           "API request ID",
        "claude_latency_s":            "LLM response latency (seconds)",
        "claude_prompt_tokens":        "Prompt token count",
        "claude_output_tokens":        "Output token count",
        "claude_cost_total_usd":       "Estimated API cost per record",
        "claude_execution_date":       "Date the scoring was run",
        "claude_is_debt_contract":     "Y/N: is this a debt contract? (99.3% Y)",
        "claude_gaap_regime":          "ASC_840 or ASC_842",
        "claude_contract_type":        "Original / Amendment / Non-debt",
        "claude_SLB_SCORE":            "Sale-leaseback covenant score (0–3)",
        "claude_SYN_SCORE":            "Synthetic lease covenant score (0–3)",
        "claude_OPL_SCORE":            "Operating lease covenant score (0–3)",
        "claude_VAR_SCORE":            "Variable lease covenant score (0–2)",
        "claude_RES_SCORE":            "Residual value covenant score (0–2)",
        "claude_GAAP_OVERRIDE_SCORE":  "GAAP override covenant score (0–2)",
        "claude_FREEZE_SCORE":         "Freeze clause covenant score (0–2)",
        "claude_evidence":             "Supporting text excerpt from filing",
    }

    # Identify which columns in the merged result came from contracts source
    contract_cols = [c for c in df.columns if c in descriptions]

    date_min = df["tranche_active_date"].min().strftime("%b %Y")
    date_max = df["tranche_active_date"].max().strftime("%b %Y")

    lines = [
        "Claude Output Dataset — Variable List",
        "Source: Ayung's RA LLM-scored dataset (contracts_dv.parquet)",
        f"Rows: {len(df):,} | Columns (contracts source): {len(contract_cols)} | Period: {date_min} – {date_max}",
        "=" * 63,
        "",
    ]

    for i, col in enumerate(contract_cols, 1):
        dtype = str(df[col].dtype)
        desc  = descriptions.get(col, "")
        lines.append(f"{i:2}. {col:<30} [{dtype:<7}]  {desc}")

    VAR_LIST_FILE.write_text("\n".join(lines) + "\n")
    print(f"Variable list written to {VAR_LIST_FILE}")

write_variable_list(result)


def write_contracts_variable_list(df: pd.DataFrame) -> None:
    """Write a full variable listing for contracts_base.parquet, grouped by source."""

    # Column groups by source — order determines section order in the output file
    groups = {
        "IDENTIFIERS & MERGE KEYS (lender experience anchor)": [
            "borrower_id", "tranche_active_date", "cik", "gvkey", "file_link",
        ],
        "LENDER EXPERIENCE — 12-MONTH WINDOW": [
            c for c in df.columns if c.endswith("_12")
        ],
        "LENDER EXPERIENCE — 24-MONTH WINDOW": [
            c for c in df.columns if c.endswith("_24")
        ],
        "LENDER EXPERIENCE — 36-MONTH WINDOW": [
            c for c in df.columns if c.endswith("_36")
        ],
        "RATINGS & EXTERNAL CONTROLS (external_variables.csv)": [
            "lpc_deal_id_extvars",
            "past_relationship_count", "number_of_lenders", "relationship_freq",
            "current_rating", "rating_date", "rating_type",
            "deal_active_date", "amendment_seq",
        ],
        "RATING RECODE (derived)": [
            "modified_rating", "num_rating", "non_rated", "rating_stale",
            "modified_rating_36m", "num_rating_36m", "non_rated_36m",
        ],
        "BASE CONTRACTS — IDENTIFIERS & METADATA (contracts_dv.parquet)": [
            "contract_id", "cname", "ftype", "fdate",
            "text_link", "html_link", "contract_file_link",
            "line1", "line2", "line3", "line4", "line5",
            "first_date", "newest_date_first20",
        ],
        "BASE CONTRACTS — LLM EXECUTION METADATA (contracts_dv.parquet)": [
            "claude_timestamp_utc", "claude_finalized_by", "claude_validation_ok",
            "claude_validation_errors", "claude_evidence_pack_chars",
            "claude_evidence_pack_hash", "claude_request_id", "claude_latency_s",
            "claude_prompt_tokens", "claude_output_tokens", "claude_cost_total_usd",
            "claude_execution_date",
        ],
        "BASE CONTRACTS — LLM RESEARCH VARIABLES (contracts_dv.parquet)": [
            "claude_is_debt_contract", "claude_gaap_regime", "claude_contract_type",
            "claude_SLB_SCORE", "claude_SYN_SCORE", "claude_OPL_SCORE",
            "claude_VAR_SCORE", "claude_RES_SCORE",
            "claude_GAAP_OVERRIDE_SCORE", "claude_FREEZE_SCORE",
            "claude_evidence",
        ],
        "ASC 842 ADOPTION (asc_adoption_counts.csv)": [
            "adoption_date", "date_source", "description",
            "pre_contract_count", "post_contract_count",
            "pre_amendment_count", "post_amendment_count",
        ],
    }

    # Catch any columns not assigned to a group
    assigned = {c for cols in groups.values() for c in cols}
    unassigned = [c for c in df.columns if c not in assigned]

    date_min = df["tranche_active_date"].min().strftime("%b %Y")
    date_max = df["tranche_active_date"].max().strftime("%b %Y")
    n = len(df)

    lines = [
        "contracts_base.parquet — Full Variable List",
        f"Rows: {n:,} | Columns: {df.shape[1]} | Period: {date_min} – {date_max}",
        "=" * 63,
    ]

    col_num = 1
    for section, cols in groups.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        lines += ["", f"{section}", "-" * len(section)]
        for col in present:
            dtype    = str(df[col].dtype)
            n_nonmiss = df[col].notna().sum()
            pct      = n_nonmiss / n * 100
            lines.append(f"{col_num:3}. {col:<40} [{dtype:<18}]  {n_nonmiss:>6,} non-missing ({pct:5.1f}%)")
            col_num += 1

    if unassigned:
        lines += ["", "OTHER (ungrouped)", "-" * 17]
        for col in unassigned:
            dtype    = str(df[col].dtype)
            n_nonmiss = df[col].notna().sum()
            pct      = n_nonmiss / n * 100
            lines.append(f"{col_num:3}. {col:<40} [{dtype:<18}]  {n_nonmiss:>6,} non-missing ({pct:5.1f}%)")
            col_num += 1

    CONTRACTS_VAR_FILE.write_text("\n".join(lines) + "\n")
    print(f"Contracts variable list written to {CONTRACTS_VAR_FILE}")

write_contracts_variable_list(result)
