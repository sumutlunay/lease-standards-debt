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
| `01_compustat.py` | Pull Compustat annual fundamentals from WRDS (1998–2025); construct size, profitability, leverage, and lease variables (with an XBRL fallback for `offbslease`) | `data/compustat_1998_2025.parquet` |
| `02_dealscan.py` | Load DealScan loan/tranche data; construct maturity, spread, and deal-size variables | `data/dealscan_raw.parquet`, `data/dealscan.parquet` |
| `03_contracts.py` | Merge lender experience, credit ratings, LLM-scored contracts, and ASC-842 adoption; define the analysis sample; recode ratings into an unrestricted (primary) set plus a 36-month restricted (`_36m`) set; pull FISD bond-activity variables (`bond_issuance_count`/`log_bond_count`, `cumulative_bond_proceeds`) and **supplement missing S&P ratings with the borrower's most recent FISD bond rating** (`num_rating_suppl_all` / `non_rated_suppl_all`, plus an issuance-restricted `_suppl_iss` variant) | `data/contracts_base.parquet` |
| `04_merge.py` | Merge DealScan and Compustat into the final analysis dataset; scale cumulative bond proceeds by total assets (`bond_proceeds_scaled = cumulative_bond_proceeds / (at × 1000)`, reconciling $-thousands vs $-millions) | `data/contracts.parquet` |
| `05_descriptives.py` | Descriptive statistics + correlation matrix for the regression variables (see `06`). Runs **before** the regression (descriptives indexed first) but independently re-derives the identical Model 7/8 estimation sample | `output/tables/descriptives.xlsx` |
| `06_rq1_determinants.py` | RQ1 determinants table — 8 models × 5 dependent variables (one worksheet each: SLB, SYN, OPL, VAR-RES, ALL) | `output/tables/rq1_determinants.xlsx` |
| `07_rq2_experience.py` | RQ2 lender-experience table — how lenders' exposure to irregularity events in their *other* borrowers shapes covenant design. 7 event families × 2 lender samples = 14 columns, repeated at 3 lookback windows (one worksheet each: `36`, `24`, `12`) | `output/tables/rq2_experience.xlsx` |
| `09b_rq2_experience_nocontrols.py` | RQ2 robustness diagnostic — the `07` table re-estimated with **no control variables** (RHS = the 4 experience test vars + FE only), on the same fixed sample as `07`, to check whether the experience effects survive on the fixed effects alone. Renamed from `07b_` (2026-07-18) to group with the diagnostics; kept as a faithful mirror of `07`'s control set but **history-only** (not re-run) | `output/tables/rq2_experience_nocontrols.xlsx` |
| `08_rq3_asc842.py` | RQ3 — how ASC 842 adoption moderates covenant design. Worksheet `comparison`: firm-level paired t-tests of contract/amendment counts pre vs post adoption. Worksheet `RQ3A`: 5 dependent-variable columns with `post_adoption` and its seven interactions | `output/tables/rq3_asc842.xlsx` |
| `09_diagnostics.py` | Diagnostic tables (each a function dispatched from `__main__`, one workbook each): **D1/D2** RQ2 distribution & RQ1↔RQ2 correlation checks; **D4** the GenAI score-decomposition robustness test — multinomial logit of each off-BS score component (SLB/SYN/OPL/VAR-RES) on the RQ1 covariates (`4.1` composite / `4.2` decomposed `accounting_policy`, no FE), plus `4.3` the RQ3 `post_adoption`-interaction decomposition (linear, full FE, reusing 08's machinery); **D5** RQ3 full-sample robustness — 08's RQ3A re-estimated with non-adopters kept as never-treated controls (`post_adoption=0`, N=10,794), in 08's exact RQ3A output format for line-by-line comparison against 08's own adopters-only RQ3A (08's base code unchanged; ~4 min) | `output/tables/diagnostic1_*.xlsx`, `diagnostic2_*.xlsx`, `diagnostic4_genai_decomp.xlsx`, `diagnostic5_rq3_fullsample.xlsx` |

### Supporting scripts

| Script | Description |
|--------|-------------|
| `figures_frequency.py` | Figure 1–3 frequency charts (reads the LLM-scored contracts directly) |
| `full_regression_v1.py`, `full_regression_v2.py`, `full_regression (log, FE, indicators, cvs).py` | Earlier RA regression scripts; kept for reference |

> Earlier single-spec regression variants (`05a`/`05b_rq1_determinants.py`) and a
> DealScan SQL reference are retired to the project's `legacy/` folder and are **not**
> part of this repository. The `05`/`06` prefixes were **swapped** (descriptives now run
> before the regression); the pre-swap `05_rq1_determinants.py` and `06_descriptives.py`
> are preserved in `legacy/` as permanent records.

---

## The 8-model determinants table (`06`)

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
add 21 deal/firm controls. Dense OLS for non-pair FE; sparse LSQR/FWL for the pair-FE
columns. Standard errors clustered by `gvkey`.

**The five determinants:** `accounting_policy` (ASC-842 GAAP-override / freeze, OR logic),
`offbslease` (off-balance-sheet lease intensity, winsorized 1%), `num_rating_suppl_all`
(credit rating 0–22, S&P supplemented with FISD bond ratings — see below), `non_rated_suppl_all`,
and `relationship_freq` (fraction of the deal's lenders with a prior 36-month borrower
relationship).

**The 21 controls** are the deal/firm controls plus two FISD bond-activity measures added in
this revision: `log_bond_count` = `log(1 + bond_issuance_count)` (logged, not winsorized) and
`bond_proceeds_scaled` = cumulative bond proceeds ÷ total assets (winsorized 1%).

### Credit-rating supplement (FISD)

To reduce the number of unrated observations, a borrower–tranche whose S&P rating is missing
(`num_rating == 0`) is filled with that borrower's most recent FISD bond rating, mapped onto the
same 1–22 scale (S&P/Fitch/Duff & Phelps on the S&P scale; Moody's via a dedicated crosswalk).
Two supplemented constructions are produced in `03`: `*_suppl_all` (unrestricted — any available
FISD rating) and `*_suppl_iss` (restricted to borrowers with `bond_issuance_count > 0`). The
S&P-only `num_rating` / `non_rated` are retained as a baseline. Models use the `*_suppl_all`
version. The `*_iss` restriction recovers roughly half the coverage gain of the unrestricted fill.

---

## The RQ2 lender-experience table (`07`)

Holds the Model 7 specification above fixed (DV = `ALL_score_dummy`; Industry×Year +
Borrower + Lender FEs; the 5 determinants + 21 controls, with the FISD-supplemented ratings;
SEs clustered by `gvkey`) and adds four test variables measuring the syndicate's exposure to
irregularity events in its lenders' *other* borrower portfolios.

For each event family, the event count is assigned to the **Top5** bucket when the lender
that experienced it is a top-5 lender and to the **Non-Top5** bucket otherwise, then
transformed as `log(1 + x)`. Crossed with borrower relatedness (unrelated / related), this
yields `NonTop5_Unrelated`, `Top5_Unrelated`, `NonTop5_Related`, `Top5_Related`. Being
logged, they are not winsorized.

**Seven event families**, each reported for all lenders and for lead arrangers only (14 columns):

| Label | Description |
|-------|-------------|
| AEC | Accounting estimate changes |
| FR | Financial restatements |
| IC | Internal control weakness (auditor or manager, whichever is higher — `max`) |
| GC | Going concern |
| LF | Late filing |
| MI | Material impairment |
| SP | S&P default |

The same table is produced at 36-, 24- and 12-month lookback windows (one worksheet each),
since the strength of the association depends materially on the window.

---

## The RQ3 / ASC 842 table (`08`)

**Worksheet `comparison`** — firm-level paired *t*-tests (with Wilcoxon signed-rank, since the
counts are zero-inflated) of contract and amendment counts before vs after each firm's ASC 842
adoption. Run on all 4,185 firms, read directly from the adoption file: these are **firm**
attributes, and running them on the loan-level sample would repeat each firm ~3.8 times and
overstate significance by orders of magnitude.

**Worksheet `RQ3A`** — the Model 7 specification plus an `amendment` control, the
`post_adoption` treatment dummy, and its interactions with `accounting_policy`,
`relationship_freq`, `fin_covenant_count`, `offbslease`, `non_rated_suppl_all`,
`num_rating_suppl_all` and `amendment`. Five columns, one per dependent variable
(SLB, SYN, OPL, VAR-RES, ALL).

```
post_adoption =  1   tranche signed on/after the borrower's ASC 842 adoption date
                 0   signed before
                NaN  firm has no adoption date  →  row dropped
```

The adoption file is left-joined upstream, so non-adopters carry a missing date. Those rows are
**dropped rather than coerced to zero** — RQ3 is estimated on adopting firms only; a zero would
pool never-adopters into the pre-period control group. The fixed effects are built *after* that
restriction so the reported FE counts match the sample estimated.

Level variables are winsorized **before** the interactions are formed, so `post × offbslease`
and `post × fin_covenant_count` inherit the winsorized parent.

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
python 05_descriptives.py                 # descriptives, indexed before the regressions
python 06_rq1_determinants.py             # ~15 min (8 models × 5 DVs)
python 07_rq2_experience.py               # ~30 min (14 columns × 3 windows)
python 09b_rq2_experience_nocontrols.py   # RQ2 robustness diagnostic — same sample, no controls (history-only)
python 08_rq3_asc842.py                   # ~3 min
python 09_diagnostics.py                  # diagnostics: D1/D2 (RQ2) + D4 (GenAI decomposition) + D5 (RQ3 never-adopter, ~5 min)
```
