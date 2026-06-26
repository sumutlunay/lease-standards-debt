"""
Pull Compustat North America fundamentals annual (funda) for 2000-2025.

Standard one-row-per-firm-year filters:
  indfmt='INDL', datafmt='STD', popsrc='D', consol='C', curcd='USD'
  Sample period: EXTRACT(YEAR FROM datadate) BETWEEN 2000 AND 2025
  Drop: missing AT or IB

Variables pulled:
  sich          — historical SIC code from comp.funda
  sic           — sich filled with current SIC from comp.company where sich is missing

Variables constructed:
  size          = log(AT)
  profitability = IB / AT
  bsfixed       = PPENT / AT
  liabilities   = (DLC + DLTT) / AT       — missing DLC/DLTT treated as 0
  offbslease    = (XRENT + MRC1 + MRC2 + MRC3 + MRC4 + MRC5 + MRCTA) / AT
                                           — missing lease components treated as 0
  logage        = log(years since first datadate for that gvkey in sample)
                  missing in the firm's first year (age = 0)
  btm           = SEQ / (PRCC_F * CSHO); missing if market cap <= 0 or SEQ missing
  capex         = CAPX / (AT - CAPX); missing if denominator <= 0 or CAPX missing
  loss          = 1 if IB < 0, else 0
  rand          = XRD / REVT; missing XRD treated as 0; missing if REVT <= 0
  divyield      = DVC / (PRCC_F * CSHO); missing if market cap <= 0 or DVC missing

Output: ../data/compustat_2000_2025.parquet
"""

from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import wrds

REPO_DIR      = Path(__file__).resolve().parent
DATA_DIR      = REPO_DIR.parent / "data"
CACHE_FILE    = DATA_DIR / "compustat_2000_2025.parquet"
VAR_LIST_FILE = REPO_DIR.parent / "documentation" / "variables_compustat.txt"


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
          AND EXTRACT(YEAR FROM datadate) BETWEEN 2000 AND 2025
          AND at IS NOT NULL
          AND ib IS NOT NULL
    """)

    # Fill missing sich with current SIC from comp.company (mirrors SAS CASE WHEN)
    company = db.raw_sql("SELECT gvkey, sic FROM comp.company")
    db.close()

    company["sic_co"] = pd.to_numeric(company["sic"], errors="coerce")
    df = df.merge(company[["gvkey", "sic_co"]], on="gvkey", how="left")
    df["sic"] = df["sich"].fillna(df["sic_co"])
    df = df.drop(columns=["sic_co"])

    n_sich    = df["sich"].notna().sum()
    n_filled  = df["sic"].notna().sum() - n_sich
    print(f"  sich non-missing: {n_sich:,}  |  filled from comp.company: {n_filled:,}  |  still missing: {df['sic'].isna().sum():,}")
    return df


def construct_variables(df: pd.DataFrame) -> pd.DataFrame:
    df["datadate"] = pd.to_datetime(df["datadate"])

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

    # Years since first appearance in Compustat for this gvkey; missing in first year (age=0)
    first_date = df.groupby("gvkey")["datadate"].transform("min")
    age_years  = (df["datadate"] - first_date).dt.days / 365.25
    df["logage"] = np.where(age_years > 0, np.log(age_years), np.nan)

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
    n_raw         = 24  # gvkey, datadate, fyear, sich, sic + 19 financial items
    n_constructed = 11
    total         = len(df.columns)

    lines = [
        "Compustat Variable List",
        f"Source: comp.funda + comp.company (WRDS)",
        f"Sample: indfmt='INDL', datafmt='STD', popsrc='D', consol='C', curcd='USD'",
        f"Period: 2000–2025 (datadate year)",
        f"Total columns: {total} ({n_raw} raw + {n_constructed} constructed)",
        f"Generated: {date.today()}",
        "=" * 80,
        "",
        "IDENTIFIERS",
        "-----------",
        "gvkey      — Compustat firm identifier",
        "datadate   — Fiscal year end date",
        "fyear      — Fiscal year",
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
        "                missing lease components treated as 0; missing if at <= 0",
        "logage        — log(years since first datadate for that gvkey in sample);",
        "                missing in the firm's first year (age = 0)",
        "btm           — seq / (prcc_f * csho); missing if market cap <= 0 or seq missing",
        "capex         — capx / (at - capx); missing if denominator <= 0 or capx missing",
        "loss          — 1 if ib < 0, else 0",
        "rand          — xrd / revt; missing xrd treated as 0 (non-R&D firms); missing if revt <= 0",
        "divyield      — dvc / (prcc_f * csho); missing if market cap <= 0 or dvc missing",
    ]

    VAR_LIST_FILE.write_text("\n".join(lines) + "\n")
    print(f"Variable list written to {VAR_LIST_FILE}")


REQUIRED_RAW_COLS = {"seq", "prcc_f", "csho", "capx", "xrd", "revt", "dvc"}


def main() -> None:
    if CACHE_FILE.exists():
        df = pd.read_parquet(CACHE_FILE)
        if not REQUIRED_RAW_COLS.issubset(df.columns):
            print("Cache missing new raw columns — re-querying WRDS...")
            df = pull_funda()
            df = construct_variables(df)
            df.to_parquet(CACHE_FILE, index=False)
            print(f"Saved to {CACHE_FILE}")
        else:
            print(f"Loading cached data from {CACHE_FILE}")
    else:
        print("Cache not found — querying WRDS...")
        df = pull_funda()
        df = construct_variables(df)
        df.to_parquet(CACHE_FILE, index=False)
        print(f"Saved to {CACHE_FILE}")

    describe_sample(df)
    print("\nFirst 5 rows:")
    print(df[["gvkey", "datadate", "fyear", "size", "profitability",
              "bsfixed", "liabilities", "offbslease"]].head())
    write_variable_list(df)


if __name__ == "__main__":
    main()
