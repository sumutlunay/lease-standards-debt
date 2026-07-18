"""
Pull Compustat North America fundamentals annual (funda) for 1998-2025.

Standard one-row-per-firm-year filters:
  indfmt='INDL', datafmt='STD', popsrc='D', consol='C', curcd='USD'
  Sample period: EXTRACT(YEAR FROM datadate) BETWEEN 1998 AND 2025
  Drop: missing AT or IB

Variables pulled:
  sich          — historical SIC code from comp.funda
  sic           — sich filled with current SIC from comp.company where sich is missing
  cik           — SEC filer ID from comp.company (for XBRL merge)

Variables constructed:
  size          = log(AT)
  profitability = IB / AT
  bsfixed       = PPENT / AT
  liabilities   = (DLC + DLTT) / AT       — missing DLC/DLTT treated as 0
  offbslease    = (XRENT + MRC1 + MRC2 + MRC3 + MRC4 + MRC5 + MRCTA) / AT
                                           — missing lease components treated as 0
                  XBRL fallback (per Ayung): when the Compustat numerator sum is 0,
                  use OperatingLeasesFutureMinimumPaymentsDue from the newest 10-K
                  for that fiscal year end (f_lease_rental_tags_04-24-26.csv, Box).
                  Matched on cik + fiscal year-end date (nearest within ±7 days to
                  handle 52/53-week fiscal years). Values converted from $ to $M.
  xbrl_oplease_due — matched XBRL value in $M (all rows where a match exists)
  offbslease_xbrl  — 1 if the XBRL fallback replaced the Compustat numerator
  logage        = log(1 + years since the firm's first datadate in the FULL funda history)
                  (not censored at the 2000 sample start; first year -> log(1) = 0)
  btm           = SEQ / (PRCC_F * CSHO); missing if market cap <= 0 or SEQ missing
  capex         = CAPX / (AT - CAPX); missing if denominator <= 0 or CAPX missing
  loss          = 1 if IB < 0, else 0
  rand          = XRD / REVT; missing XRD treated as 0; missing if REVT <= 0
  divyield      = DVC / (PRCC_F * CSHO); missing if market cap <= 0 or DVC missing

Output: ../data/compustat_1998_2025.parquet
"""

from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import wrds

REPO_DIR      = Path(__file__).resolve().parent
DATA_DIR      = REPO_DIR.parent / "data"
CACHE_FILE    = DATA_DIR / "compustat_1998_2025.parquet"
VAR_LIST_FILE = REPO_DIR.parent / "documentation" / "variables_compustat.txt"

# XBRL lease tags (Ayung, 04-24-26 vintage) — supplement for offbslease numerator
XBRL_CACHE    = DATA_DIR / "f_lease_rental_tags_04-24-26.parquet"
XBRL_BOX_URL  = "https://ucdavis.box.com/shared/static/3w2zv9pbcduux9ajggguk3ea8sua07d8.csv"
XBRL_TAG      = "OperatingLeasesFutureMinimumPaymentsDue"


def pull_funda() -> pd.DataFrame:
    db = wrds.Connection(wrds_username="sunay")

    df = db.raw_sql("""
        SELECT
            gvkey, datadate, fyear,
            sich,
            at, ib, ppent,
            dlc, dltt,
            xrent, mrc1, mrc2, mrc3, mrc4, mrc5, mrcta,
            seq, prcc_f, csho,
            capx,
            xrd, revt,
            dvc
        FROM comp.funda
        WHERE indfmt  = 'INDL'
          AND datafmt = 'STD'
          AND popsrc  = 'D'
          AND consol  = 'C'
          AND curcd   = 'USD'
          AND EXTRACT(YEAR FROM datadate) BETWEEN 1998 AND 2025
          AND at IS NOT NULL
          AND ib IS NOT NULL
    """)

    # Fill missing sich with current SIC from comp.company (mirrors SAS CASE WHEN)
    company = db.raw_sql("SELECT gvkey, sic, cik FROM comp.company")

    # A3: firm age must not be left-censored at the 2000 sample start. Pull each gvkey's
    # TRUE first fiscal year end from the full funda history (same standard filters, no year
    # restriction), so a firm that first appeared before 2000 gets its real age.
    first_dt = db.raw_sql("""
        SELECT gvkey, MIN(datadate) AS first_datadate
        FROM comp.funda
        WHERE indfmt  = 'INDL'
          AND datafmt = 'STD'
          AND popsrc  = 'D'
          AND consol  = 'C'
          AND curcd   = 'USD'
          AND at IS NOT NULL
          AND ib IS NOT NULL
        GROUP BY gvkey
    """)
    db.close()

    company["sic_co"] = pd.to_numeric(company["sic"], errors="coerce")
    company["cik"]    = pd.to_numeric(company["cik"], errors="coerce").astype("Int64")
    df = df.merge(company[["gvkey", "sic_co", "cik"]], on="gvkey", how="left")
    df["sic"] = df["sich"].fillna(df["sic_co"])
    df = df.drop(columns=["sic_co"])

    # Attach true first-appearance date (full history) for the un-censored firm-age in
    # construct_variables; dropped again there so it does not enter the output.
    df = df.merge(first_dt, on="gvkey", how="left")

    n_sich    = df["sich"].notna().sum()
    n_filled  = df["sic"].notna().sum() - n_sich
    print(f"  sich non-missing: {n_sich:,}  |  filled from comp.company: {n_filled:,}  |  still missing: {df['sic'].isna().sum():,}")
    return df


def construct_variables(df: pd.DataFrame) -> pd.DataFrame:
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["first_datadate"] = pd.to_datetime(df["first_datadate"])   # true first appearance (A3)

    # Base: replace AT <= 0 with NaN so all ratios propagate NaN for those rows
    at_positive = df["at"].where(df["at"] > 0)

    df["size"]          = np.log(at_positive)
    df["profitability"] = df["ib"] / at_positive
    df["bsfixed"]       = df["ppent"] / at_positive

    # Missing debt components treated as 0: firms with no reported DLC/DLTT have no debt
    df["liabilities"] = (
        df["dlc"].fillna(0) + df["dltt"].fillna(0)
    ) / at_positive

    # Missing lease components treated as 0: firms not reporting MRC/XRENT have no operating leases
    lease_cols = ["xrent", "mrc1", "mrc2", "mrc3", "mrc4", "mrc5", "mrcta"]
    df["offbslease"] = df[lease_cols].fillna(0).sum(axis=1) / at_positive

    # A3: firm age from the TRUE first fiscal year end (full funda history, not censored at the
    # 2000 sample start). log(1 + age) so the firm's first year (age = 0) maps to 0 instead of
    # dropping via log(0). Fall back to the in-sample first datadate if a gvkey somehow lacks a
    # full-history date.
    insample_first = df.groupby("gvkey")["datadate"].transform("min")
    first_date     = df["first_datadate"].fillna(insample_first)
    age_years      = (df["datadate"] - first_date).dt.days / 365.25
    df["logage"]   = np.log1p(age_years.clip(lower=0))   # log(1 + age)
    df = df.drop(columns="first_datadate")

    # Book-to-market: SEQ / (PRCC_F * CSHO); missing if market cap <= 0 or SEQ missing
    mktcap = (df["prcc_f"] * df["csho"]).where(df["prcc_f"] * df["csho"] > 0)
    df["btm"] = df["seq"] / mktcap

    # Capex ratio: CAPX / (AT - CAPX); missing if denominator <= 0 or CAPX missing
    denom = (df["at"] - df["capx"]).where((df["at"] - df["capx"]) > 0)
    df["capex"] = df["capx"] / denom

    # Loss indicator: 1 if IB < 0, else 0 (IB is never missing — filtered at query level)
    df["loss"] = (df["ib"] < 0).astype(int)

    # R&D intensity: XRD / REVT; missing XRD treated as 0 (non-R&D firms); missing if REVT <= 0
    revt_pos = df["revt"].where(df["revt"] > 0)
    df["rand"] = df["xrd"].fillna(0) / revt_pos

    # Dividend yield: DVC / (PRCC_F * CSHO); missing if market cap <= 0 or DVC missing
    df["divyield"] = df["dvc"] / mktcap

    return df


def load_xbrl_lease() -> pd.DataFrame:
    """Load XBRL OperatingLeasesFutureMinimumPaymentsDue, one row per cik + FY end.

    Filters to the 10-K family (10-K, 10-K/A, 10-KT, 10-KT/A) and keeps the value
    from the newest filing for each cik + fiscal-year-end. Verified: within a
    cik + FY-end + filing date there is never more than one distinct value, so
    this dedup is exact. Dollar values converted to $ millions to match Compustat.
    """
    if XBRL_CACHE.exists():
        xbrl = pd.read_parquet(XBRL_CACHE)
    else:
        print(f"Downloading XBRL lease tags from Box …")
        xbrl = pd.read_csv(XBRL_BOX_URL)
        xbrl.to_parquet(XBRL_CACHE, index=False)

    t = xbrl[(xbrl["Tag Title"] == XBRL_TAG)
             & xbrl["Form Type"].str.startswith("10-K")].copy()
    t["fyend"] = pd.to_datetime(t["Fiscal Year-End Date"], format="mixed")
    t["fdate"] = pd.to_datetime(t["Form Filing Date"], format="mixed")

    # Newest 10-K per cik + fiscal year end
    t = t.sort_values("fdate").drop_duplicates(subset=["CIK", "fyend"], keep="last")

    t["xbrl_oplease_due"] = t["Dollar Value"] / 1e6
    out = t[["CIK", "fyend", "xbrl_oplease_due"]].rename(columns={"CIK": "cik"})
    out["cik"] = out["cik"].astype("Int64")
    print(f"  XBRL: {len(out):,} cik × fiscal-year-end values for {XBRL_TAG}")
    return out


def apply_xbrl_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Replace offbslease numerator with XBRL value where Compustat reports 0.

    Ayung's rule: if XRENT + MRC1–5 + MRCTA is missing or zero (missing is already
    filled to 0 upstream), use OperatingLeasesFutureMinimumPaymentsDue instead.
    Match on cik + fiscal year-end via nearest date within ±7 days — exact for
    calendar month-end firms, and catches 52/53-week firms whose XBRL period end
    differs from Compustat's month-end datadate by a few days.
    """
    xbrl = load_xbrl_lease()

    # merge_asof needs sorted keys and no missing join values on the left
    df["_row_order"] = range(len(df))
    left  = df[df["cik"].notna()].sort_values("datadate")
    right = xbrl.sort_values("fyend")

    merged = pd.merge_asof(
        left, right,
        left_on="datadate", right_on="fyend",
        by="cik",
        direction="nearest",
        tolerance=pd.Timedelta(days=7),
    ).drop(columns=["fyend"])

    no_cik = df[df["cik"].isna()].copy()
    no_cik["xbrl_oplease_due"] = np.nan
    df = (
        pd.concat([merged, no_cik])
        .sort_values("_row_order")
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )

    lease_cols = ["xrent", "mrc1", "mrc2", "mrc3", "mrc4", "mrc5", "mrcta"]
    lease_num  = df[lease_cols].fillna(0).sum(axis=1)
    at_positive = df["at"].where(df["at"] > 0)

    use_xbrl = (lease_num == 0) & df["xbrl_oplease_due"].notna() & at_positive.notna()
    df["offbslease_xbrl"] = use_xbrl.astype(int)
    df.loc[use_xbrl, "offbslease"] = df.loc[use_xbrl, "xbrl_oplease_due"] / at_positive[use_xbrl]

    n_matched = df["xbrl_oplease_due"].notna().sum()
    print(f"\n── XBRL offbslease fallback ──────────────────────────────────────────")
    print(f"  Firm-years matched to XBRL:      {n_matched:>7,}")
    print(f"  Compustat numerator == 0:        {(lease_num == 0).sum():>7,}")
    print(f"  offbslease replaced by XBRL:     {use_xbrl.sum():>7,}")
    return df


def describe_sample(df: pd.DataFrame) -> None:
    print(f"\nTotal rows: {len(df):,}")
    print(f"Unique firms (gvkey): {df['gvkey'].nunique():,}")

    print("\nRows by fiscal year:")
    print(df["fyear"].value_counts().sort_index().to_string())

    n_sic = df["sic"].notna().sum()
    print(f"\nsic coverage: {n_sic:,} / {len(df):,} rows non-missing  ({n_sic / len(df) * 100:.1f}%)")
    constructed = ["size", "profitability", "bsfixed", "liabilities", "offbslease", "logage", "btm", "capex", "loss", "rand", "divyield"]
    print("\nNon-missing counts and basic stats on constructed variables:")
    print(df[constructed].describe().round(3).to_string())


def write_variable_list(df: pd.DataFrame) -> None:
    n_raw         = 26  # gvkey, datadate, fyear, sich, sic, cik + 19 financial items + xbrl_oplease_due
    n_constructed = 12
    total         = len(df.columns)

    lines = [
        "Compustat Variable List",
        f"Source: comp.funda + comp.company (WRDS)",
        f"Sample: indfmt='INDL', datafmt='STD', popsrc='D', consol='C', curcd='USD'",
        f"Period: 1998–2025 (datadate year)",
        f"Total columns: {total} ({n_raw} raw + {n_constructed} constructed)",
        f"Generated: {date.today()}",
        "=" * 80,
        "",
        "IDENTIFIERS",
        "-----------",
        "gvkey      — Compustat firm identifier",
        "datadate   — Fiscal year end date",
        "fyear      — Fiscal year",
        "cik        — SEC filer ID (from comp.company; merge key for XBRL supplement)",
        "",
        "SIC CODE",
        "--------",
        "sich       — Historical SIC code (from comp.funda)",
        "sic        — sich filled with current SIC from comp.company where sich is missing",
        "",
        "RAW FINANCIALS (from comp.funda)",
        "---------------------------------",
        "at         — Total assets",
        "ib         — Income before extraordinary items",
        "ppent      — Net property, plant & equipment",
        "dlc        — Debt in current liabilities",
        "dltt       — Long-term debt total",
        "xrent      — Rental expense",
        "mrc1       — Minimum rental commitment, year 1",
        "mrc2       — Minimum rental commitment, year 2",
        "mrc3       — Minimum rental commitment, year 3",
        "mrc4       — Minimum rental commitment, year 4",
        "mrc5       — Minimum rental commitment, year 5",
        "mrcta      — Minimum rental commitment, total amount",
        "seq        — Stockholders' equity total",
        "prcc_f     — Price close, fiscal year end",
        "csho       — Common shares outstanding",
        "capx       — Capital expenditures",
        "xrd        — Research and development expense",
        "revt       — Total revenue",
        "dvc        — Dividends common/ordinary",
        "",
        "CONSTRUCTED VARIABLES",
        "---------------------",
        "size          — log(at); missing if at <= 0",
        "profitability — ib / at; missing if at <= 0",
        "bsfixed       — ppent / at; missing if at <= 0",
        "liabilities   — (dlc + dltt) / at; missing dlc/dltt treated as 0; missing if at <= 0",
        "offbslease    — (xrent + mrc1 + mrc2 + mrc3 + mrc4 + mrc5 + mrcta) / at;",
        "                missing lease components treated as 0; missing if at <= 0;",
        "                if numerator sum == 0, falls back to XBRL",
        "                OperatingLeasesFutureMinimumPaymentsDue (newest 10-K, $M) / at",
        "xbrl_oplease_due — XBRL OperatingLeasesFutureMinimumPaymentsDue in $M,",
        "                newest 10-K per cik + FY end (f_lease_rental_tags_04-24-26, Box)",
        "offbslease_xbrl — 1 if offbslease numerator came from XBRL fallback, else 0",
        "logage        — log(1 + years since the firm's first datadate in full funda history);",
        "                un-censored at 2000; first year -> log(1) = 0",
        "btm           — seq / (prcc_f * csho); missing if market cap <= 0 or seq missing",
        "capex         — capx / (at - capx); missing if denominator <= 0 or capx missing",
        "loss          — 1 if ib < 0, else 0",
        "rand          — xrd / revt; missing xrd treated as 0 (non-R&D firms); missing if revt <= 0",
        "divyield      — dvc / (prcc_f * csho); missing if market cap <= 0 or dvc missing",
    ]

    VAR_LIST_FILE.write_text("\n".join(lines) + "\n")
    print(f"Variable list written to {VAR_LIST_FILE}")


REQUIRED_RAW_COLS = {"seq", "prcc_f", "csho", "capx", "xrd", "revt", "dvc",
                     "cik", "xbrl_oplease_due", "offbslease_xbrl"}


def build() -> pd.DataFrame:
    df = pull_funda()
    df = construct_variables(df)
    df = apply_xbrl_fallback(df)
    df.to_parquet(CACHE_FILE, index=False)
    print(f"Saved to {CACHE_FILE}")
    return df


def main() -> None:
    if CACHE_FILE.exists():
        df = pd.read_parquet(CACHE_FILE)
        if not REQUIRED_RAW_COLS.issubset(df.columns):
            print("Cache missing new raw columns — re-querying WRDS...")
            df = build()
        else:
            print(f"Loading cached data from {CACHE_FILE}")
    else:
        print("Cache not found — querying WRDS...")
        df = build()

    describe_sample(df)
    print("\nFirst 5 rows:")
    print(df[["gvkey", "datadate", "fyear", "size", "profitability",
              "bsfixed", "liabilities", "offbslease"]].head())
    write_variable_list(df)


if __name__ == "__main__":
    main()
