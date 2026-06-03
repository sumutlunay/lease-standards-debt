"""
Pull Compustat North America fundamentals annual (funda) for 2000-2025.

Standard one-row-per-firm-year filters:
  indfmt='INDL', datafmt='STD', popsrc='D', consol='C', curcd='USD'
  Sample period: EXTRACT(YEAR FROM datadate) BETWEEN 2000 AND 2025
  Drop: missing AT or IB

Variables constructed:
  size          = log(AT)
  profitability = IB / AT
  bsfixed       = PPENT / AT
  liabilities   = (DLC + DLTT) / AT       — missing DLC/DLTT treated as 0
  offbslease    = (XRENT + MRC1 + MRC2 + MRC3 + MRC4 + MRC5 + MRCTA) / AT
                                           — missing lease components treated as 0

Output: ../data/compustat_2000_2025.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
import wrds

REPO_DIR   = Path(__file__).resolve().parent
CACHE_FILE = REPO_DIR.parent / "data" / "compustat_2000_2025.parquet"


def pull_funda() -> pd.DataFrame:
    db = wrds.Connection(wrds_username="sunay")

    query = """
        SELECT
            gvkey, datadate, fyear,
            at, ib, ppent,
            dlc, dltt,
            xrent, mrc1, mrc2, mrc3, mrc4, mrc5, mrcta
        FROM comp.funda
        WHERE indfmt  = 'INDL'
          AND datafmt = 'STD'
          AND popsrc  = 'D'
          AND consol  = 'C'
          AND curcd   = 'USD'
          AND EXTRACT(YEAR FROM datadate) BETWEEN 2000 AND 2025
          AND at IS NOT NULL
          AND ib IS NOT NULL
    """
    df = db.raw_sql(query)
    db.close()
    return df


def construct_variables(df: pd.DataFrame) -> pd.DataFrame:
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

    return df


def describe_sample(df: pd.DataFrame) -> None:
    print(f"\nTotal rows: {len(df):,}")
    print(f"Unique firms (gvkey): {df['gvkey'].nunique():,}")

    print("\nRows by fiscal year:")
    print(df["fyear"].value_counts().sort_index().to_string())

    constructed = ["size", "profitability", "bsfixed", "liabilities", "offbslease"]
    print("\nNon-missing counts and basic stats on constructed variables:")
    print(df[constructed].describe().round(3).to_string())


def main() -> None:
    if CACHE_FILE.exists():
        print(f"Loading cached data from {CACHE_FILE}")
        df = pd.read_parquet(CACHE_FILE)
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


if __name__ == "__main__":
    main()
