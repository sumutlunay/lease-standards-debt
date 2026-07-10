# Lease Standards & Debt Contracting — Analysis Code

Research code for an archival accounting study of how lease accounting standards
(ASC 842) affect the design of debt contractual features — specifically
off-balance-sheet lease covenants in syndicated loans.

**Authors:** Sunay Mutlu (Kennesaw State University) · Ayung Tseng (UC Davis)

> **Note:** This repository contains **only the analysis code**. The underlying data
> (WRDS/Compustat, DealScan, LLM-scored contracts, ratings) is proprietary and is
> **not** included. Scripts expect cached `data/*.parquet` files produced by the
> pipeline and read/write relative to the project root (`../data`, `../output`).

---

## Pipeline

Run in numbered order; each stage caches its output as parquet for the next.

| Script | Description | Output |
|--------|-------------|--------|
| `01_compustat.py` | Pull Compustat annual fundamentals from WRDS (2000–2025); construct size, profitability, leverage, and lease variables (with an XBRL fallback for `offbslease`) | `data/compustat_2000_2025.parquet` |
| `02_dealscan.py` | Load DealScan loan/tranche data; construct maturity, spread, and deal-size variables | `data/dealscan_raw.parquet`, `data/dealscan.parquet` |
| `03_contracts.py` | Merge lender experience, credit ratings, LLM-scored contracts, and ASC-842 adoption; define the analysis sample; recode ratings into an unrestricted (primary) set plus a 36-month restricted (`_36m`) set | `data/contracts_base.parquet` |
| `04_merge.py` | Merge DealScan and Compustat into the final analysis dataset | `data/contracts.parquet` |
| `05_rq1_determinants.py` | RQ1 determinants table — 8 models × 5 dependent variables (one worksheet each: SLB, SYN, OPL, VAR-RES, ALL) | `output/tables/rq1_determinants.xlsx` |
| `06_descriptives.py` | Descriptive statistics + correlation matrix for the 05 variables | `output/tables/descriptives.xlsx` |
| `07_rq2_experience.py` | RQ2 lender-experience table — how lenders' exposure to irregularity events in their *other* borrowers shapes covenant design. 8 event families × 2 lender samples = 16 columns, repeated at 3 lookback windows (one worksheet each: `36`, `24`, `12`) | `output/tables/rq2_experience.xlsx` |

### Supporting scripts

| Script | Description |
|--------|-------------|
| `figures_frequency.py` | Figure 1–3 frequency charts (reads the LLM-scored contracts directly) |
| `full_regression_v1.py`, `full_regression_v2.py`, `full_regression (log, FE, indicators, cvs).py` | Earlier RA regression scripts; kept for reference |

> Earlier single-spec regression variants (`05a`/`05b_rq1_determinants.py`) and a
> DealScan SQL reference are retired to the project's `legacy/` folder and are **not**
> part of this repository.

---

## The 8-model determinants table (`05`)

Each dependent variable is a dummy = 1 if **any** listed `claude_*_SCORE > 0`, else 0:

| Sheet | Dependent variable |
|-------|--------------------|
| SLB | `claude_SLB_SCORE` (sale-leaseback) |
| SYN | `claude_SYN_SCORE` (synthetic lease) |
| OPL | `claude_OPL_SCORE` (operating lease) |
| VAR-RES | `claude_VAR_SCORE` **or** `claude_RES_SCORE` |
| ALL | any of SLB / SYN / OPL / VAR / RES |

**Fixed-effect structure** (built up incrementally across the 8 columns):
Industry×Year (2-digit SIC × year) → + Borrower → + Lender (multi-hot) →
Borrower-Lender Pair (multi-hot). Columns 5–6 add the five determinants; columns 7–8
add 19 deal/firm controls. Dense OLS for non-pair FE; sparse LSQR/FWL for the pair-FE
columns. Standard errors clustered by `gvkey`.

**The five determinants:** `accounting_policy` (ASC-842 GAAP-override / freeze, OR logic),
`offbslease` (off-balance-sheet lease intensity, winsorized 1%), `num_rating` (S&P
long-term rating 0–22), `non_rated`, and `relationship_freq` (fraction of the deal's
lenders with a prior 36-month borrower relationship).

---

## The RQ2 lender-experience table (`07`)

Holds the Model 7 specification above fixed (DV = `ALL_score_dummy`; Industry×Year +
Borrower + Lender FEs; the 5 determinants + 19 controls; SEs clustered by `gvkey`) and adds
four test variables measuring the syndicate's exposure to irregularity events in its
lenders' *other* borrower portfolios.

For each event family, the event count is assigned to the **Top5** bucket when the lender
that experienced it is a top-5 lender and to the **Non-Top5** bucket otherwise, then
transformed as `log(1 + x)`. Crossed with borrower relatedness (unrelated / related), this
yields `NonTop5_Unrelated`, `Top5_Unrelated`, `NonTop5_Related`, `Top5_Related`. Being
logged, they are not winsorized.

**Eight event families**, each reported for all lenders and for lead arrangers only:

| Label | Description |
|-------|-------------|
| AEC | Accounting estimate changes |
| FR | Financial restatements |
| A_IC | Internal control weakness identified by auditor |
| M_IC | Internal control weakness identified by manager |
| GC | Going concern |
| LF | Late filing |
| MI | Material impairment |
| SP | S&P default |

The same table is produced at 36-, 24- and 12-month lookback windows (one worksheet each),
since the strength of the association depends materially on the window.

---

## Environment

Python 3.12. WRDS access via the `wrds` package (credentials in `~/.pgpass`).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install wrds pandas pyarrow scipy statsmodels xlsxwriter openpyxl

# Run the pipeline in order
python 01_compustat.py
python 02_dealscan.py
python 03_contracts.py
python 04_merge.py
python 05_rq1_determinants.py
python 06_descriptives.py
python 07_rq2_experience.py   # ~30 min (48 regressions)
```
