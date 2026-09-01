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

## Related repositories and external resources

The data behind this study is not in this repository (see the note above). These are the
external components a reader needs in order to follow how it was assembled.

| Resource | Role |
|---|---|
| [`CS-DS-Linktable`](https://github.com/sumutlunay/CS-DS-Linktable) | DealScan–Compustat borrower crosswalk (DealScan `borrower_id` → Compustat `gvkey`). The table itself is [`cs_ds_linktable.csv`](https://github.com/sumutlunay/CS-DS-Linktable/blob/main/cs_ds_linktable.csv); [`cs_ds_linktable_detail.csv`](https://github.com/sumutlunay/CS-DS-Linktable/blob/main/cs_ds_linktable_detail.csv) carries the match detail and [`cs_ds_linktable.ipynb`](https://github.com/sumutlunay/CS-DS-Linktable/blob/main/cs_ds_linktable.ipynb) the construction. Applied **upstream** of this repository: `gvkey` and `cik` arrive already attached to the lender-experience files that `03_contracts.py` reads, so no script here calls the table directly. Forked (MIT-licensed) from [`DarrenTheLamb01/CS-DS-Linktable`](https://github.com/DarrenTheLamb01/CS-DS-Linktable) so the version used here stays available. ⚠ A borrower can map to more than one `gvkey`; the duplicates are resolved upstream. |

---

## Repository layout

```
code/
├── 01_compustat.py … 04_merge.py   the data build — run in order, each caches parquet
├── manuscript/                     the paper's numbered exhibits (Tables 1–6, Figures 1–2)
├── exploratory/                    specification variants, superset analyses, diagnostics
└── initial/                        the RA's original scripts; reference only, not maintained
```

Everything below `code/` sits one directory deep and resolves the project root as
`Path(__file__).resolve().parent.parent.parent`, so `../data` and `../output` behave
identically from any subfolder. Scripts in `initial/` are the exception: they take their
paths as command-line arguments resolved against the **current working directory**.

---

## Pipeline (`code/`)

The data build. Run in numbered order; each stage caches its output as parquet for the next.

| Script | Description | Output |
|--------|-------------|--------|
| `01_compustat.py` | Pull Compustat annual fundamentals from WRDS (1998–2025); construct size, profitability, leverage, and lease variables (with an XBRL fallback for `offbslease`) | `data/compustat_1998_2025.parquet` |
| `02_dealscan.py` | Load DealScan loan/tranche data; construct maturity, spread, and deal-size variables | `data/dealscan_raw.parquet`, `data/dealscan.parquet` |
| `03_contracts.py` | Merge lender experience, credit ratings, LLM-scored contracts, and ASC-842 adoption; define the analysis sample; recode ratings, supplement missing S&P ratings from FISD, and build the credit-quality **bucket** indicators (`ig_grade`, `BB_grade`, `B_grade`, `CCC_below`) plus `amendment` / `amendment_claude` | `data/contracts_base.parquet` |
| `04_merge.py` | Merge DealScan and Compustat into the final analysis dataset; scale cumulative bond proceeds by total assets | `data/fulldata.parquet` |

Everything downstream reads `data/fulldata.parquet` and can be run independently, in any
order, once `04` has completed.

---

## `manuscript/` — the paper's exhibits

One script per numbered exhibit. All report **centered** R²/Adj. R², **gvkey-clustered**
standard errors, and **coefficients + significance stars only, no t-statistics**. Rows carry
the **original variable names**, not manuscript labels.

| Script | Output | Contents |
|--------|--------|----------|
| `table1A_descriptives.py` | `table1A.xlsx` | Table 1 Panel A — Mean/P25/P50/P75/Std for the 35 analysis variables. N = 11,184 |
| `table1B_experience.py` | `table1B.xlsx` | Panel B — P10–P90 for the 28 RQ2 lender-experience variables, at the 12- and 36-month windows (56 rows). N = 11,184 |
| `table1C_correlations.py` | `table1C.xlsx` | Panel C — 8×8 Spearman correlation matrix over the five recognition dummies and the three accounting-policy variables. N = 11,184 |
| `table2_fe_decomposition.py` | `table2.xlsx` | Table 2 — FE-only variance decomposition, 8 FE structures × 5 DV sheets, no regressors. Full scored sample, N = 14,584 |
| `table3_determinants.py` | `table3.xlsx` | Table 3 — RQ1 Model 7 (IY + Borrower + Lender FE, determinants + 20 controls), 5 DVs as columns. N = 11,184 |
| `table4_rq2_nocontrols.py` | `table4.xlsx` | Table 4 — RQ2 12-month experience, **no controls** (4 test variables + FE). Two panels (all / lead lenders) × 7 event families. N = 11,184 / 11,178 |
| `table5_rq2_controls.py` | `table5.xlsx` | Table 5 — Table 4 **with** the full control set (estimated, not tabulated). Same sample, so 4 and 5 differ only in the RHS |
| `table6_rq3_asc842.py` | `table6.xlsx` | Table 6 — RQ3 Eq. (3) on symmetric windows around adoption, adopting firms only. **Four sheets:** Panel A ±3y (N = 1,704) and Panel B ±5y (N = 2,602) with composite `accounting_policy`; Panels C/D repeat each window with `accounting_policy` **decomposed** into `gaap_override` + `freeze`, both as main effects and `post ×` interactions |
| `table6_supplement.py` | `table6_supplement.xlsx` | Table 6 supplement — six sheets, two per window: a correlation matrix, untreated-cell score-level counts, and an `accounting_policy` × `post_adoption` 2×2 panel stacking contract counts over cell means of `ALL_score_dummy`, with the policy 1−0 gap per column (pooled two-proportion z) and a DiD column differencing the post and pre gaps (unpooled z; neither clusters on `gvkey`, unlike the tables). Asserts its samples match Table 6's panels |
| `figure1.py` | `figure1.png` | Figure 1 — contract frequency by year and GenAI score category, % share within year, 7 panels. Reads the scored corpus (N = 18,760), **not** the estimation sample |
| `figure2.py` | `figure2.png` | Figure 2 — RQ3 event study, `accounting_policy × event time` ± 95% CI, on the **±3-year** window so it matches Table 6 Panel A row-for-row |

### Credit-quality specification

Credit quality enters every table as four mutually-exclusive **bucket dummies** rather than a
linear 0–22 scale. All are built in `03` on `num_rating_suppl_all` (the S&P rating supplemented
with the borrower's most recent FISD bond rating):

| | |
|---|---|
| `ig_grade` | investment grade, BBB− or above (≥ 13) — **OMITTED REFERENCE**, never a regressor |
| `BB_grade` | BB+ / BB / BB− (10–12) |
| `B_grade` | B+ / B / B− (7–9) |
| `CCC_below` | CCC+ and below, incl. CC, C, D (1–6) |
| `non_rated_suppl_all` | still unrated after the supplement (= 0) |

The linear `num_rating_suppl_all`, the S&P-only `num_rating`, and the issuance-restricted
`num_rating_suppl_iss` all remain in `fulldata.parquet` as robustness baselines; nothing
reads them as a regressor.

---

## `exploratory/` — variants, supersets, diagnostics

Self-contained scripts kept alongside the pipeline. Each reads `data/fulldata.parquet` and
writes its own workbook; none import from each other or from `manuscript/`.

| Script | Description |
|--------|-------------|
| `05_descriptives.py` | Descriptives + correlation matrix over the full variable set — the superset of Table 1 Panel A (ten statistics, plus a correlation sheet) |
| `06_rq1_determinants.py` | RQ1 determinants — **8 models × 5 DVs** with incremental FE, of which Table 3 reports Model 7 only |
| `07_rq2_experience.py` | RQ2 lender experience — 7 event families × 2 lender samples at **three** lookback windows (36/24/12), of which Tables 4–5 report the 12-month |
| `08_rq3_asc842.py` | RQ3 on the **full** adopting-firm sample (no window restriction), plus the firm-level `comparison` sheet of pre/post contract and amendment counts |
| `09_diagnostics.py` | D1/D2 RQ2 distributions and RQ1↔RQ2 correlations; D4 the GenAI score decomposition; D5 the RQ3 never-adopter robustness check |
| `06b`–`06d`, `07b`–`07c`, `08b`–`08d` | The manuscript-table drafts these exhibits were developed from, including the linear-rating and `amendment_claude` variants |
| `09b_rq2_experience_nocontrols.py` | RQ2 no-controls robustness on the fixed `07` sample — **history only**, not re-run |
| `figures_frequency.py` | The three-figure original that `manuscript/figure1.py` was isolated from |
| `figures_rq3_eventstudy.py` | The event study on the **±5-year** window (matches Table 6 Panel B) |

---

## `initial/`

The RA's original regression scripts, superseded by the pipeline and kept for reference only.
They are argparse CLIs that read from Box at runtime and write wherever `--out` points
(default: relative to the current working directory).

| Script | Description |
|--------|-------------|
| `full_regression_v1.py` | FE-A / FE-B specifications; full and `_lean` workbooks |
| `full_regression_v2.py` | FE-C / FE-D specifications; full and `_lean` workbooks |
| `full_regression (log, FE, indicators, cvs).py` | Earlier combined variant |

> Earlier single-spec variants (`05a`/`05b_rq1_determinants.py`), the pre-swap
> `05_rq1_determinants.py` / `06_descriptives.py`, the retired linear-rating outputs and a
> DealScan SQL reference are archived in the project's `legacy/` folder and are **not** part
> of this repository.

---

## Specification notes

### Fixed effects

Industry×Year (2-digit SIC × year) + Borrower + Lender, the last multi-hot because a contract
has many lenders. Table 2 additionally decomposes across Lead-arranger and Lead-left
definitions. Models are fit with **no separate intercept** — the complete Industry×Year block
already spans the constant — which is why R² is reported centered.

### The RQ1 determinants

`accounting_policy` (ASC-842 GAAP-override OR freeze), `offbslease` (off-balance-sheet lease
intensity, winsorized 1%), the four credit-quality buckets above, and `relationship_freq`
(fraction of the deal's lenders with a prior 36-month borrower relationship). Plus 20
deal/firm controls, including two FISD bond-activity measures.

⚠ **`accounting_policy` is very nearly `gaap_override`.** It is the OR of `gaap_override` and
`freeze`, but across the full scored sample only **1 contract of 14,584** has freeze without
GAAP override, so the two are identical for all practical purposes and correlate at 1.00. The
Table 6 Panels C/D decomposition should be read with that in mind: `freeze` is the incremental
effect for contracts that also freeze the standard, not a standalone effect.

### The RQ2 test variables

For each event family, the event count is assigned to the **Top5** bucket when the lender that
experienced it is a top-5 lender and to **Non-Top5** otherwise, then transformed `log(1+x)`.
Crossed with borrower relatedness this yields `NonTop5_Unrelated`, `Top5_Unrelated`,
`NonTop5_Related`, `Top5_Related`. Being logged, they are not winsorized.

Seven event families: **AEC** accounting estimate changes · **FR** financial restatements ·
**IC** internal control weakness (auditor or manager, whichever is higher) · **GC** going
concern · **LF** late filing · **MI** material impairment · **SP** S&P default.

### RQ3 sample construction

```
post_adoption =  1   tranche signed on/after the borrower's ASC 842 adoption date
                 0   signed before
                NaN  firm has no adoption date  →  row dropped
```

Non-adopters are **dropped, not coerced to zero** — a zero would pool never-adopters into the
pre-period control group. Fixed effects are built after that restriction so the reported FE
counts match the estimated sample. Level variables are winsorized **before** the interactions
are formed, so `post × offbslease` inherits the winsorized parent.

---

## Environment

Python 3.12. WRDS access via the `wrds` package (credentials in `~/.pgpass`).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install wrds pandas pyarrow scipy statsmodels matplotlib xlsxwriter openpyxl

# 1. Build the data (in order)
python 01_compustat.py
python 02_dealscan.py
python 03_contracts.py
python 04_merge.py

# 2. Produce the exhibits (any order, once 04 has run)
python manuscript/table1A_descriptives.py
python manuscript/table3_determinants.py
python manuscript/table6_rq3_asc842.py
python manuscript/figure2.py
# … etc
```

Runtimes vary: the Table 1 panels and the supplement take seconds; Tables 2–5 build large
multi-hot FE matrices and take roughly 5–20 minutes each.
