"""
Pull DealScan loan-tranche data from Ayung's Box share and construct variables.

Source: Dealscan_1981_2025_full.csv (shared by Dr. Ayung Tseng, UC Davis)
Box direct link: https://ucdavis.box.com/shared/static/pkcfg63i84lqfza5qm1jlnne8yzmcqvy.csv

Two-file cache strategy:
  dealscan_raw.parquet  — raw CSV converted to parquet; downloaded once, never regenerated
                          unless the file is deleted (e.g. when Ayung updates the source).
  dealscan.parquet      — constructed variables built on top of raw; can be rebuilt at any
                          time in seconds without re-downloading from Box.

Variables constructed:
  maturity          = log(months between tranche_active_date and tranche_maturity_date)
                      months computed as calendar days / (365.25 / 12)
                      set to missing if either date is missing, or if the difference <= 0
  log_lender_count  = log(number_of_lenders); set to missing if number_of_lenders <= 0
  log_interest      = log(all_in_spread_drawn_bps); set to missing if spread <= 0
  log_deal_amount   = log(deal_amount_converted); set to missing if amount <= 0
  perf_pricing      = 1 if performance_pricing is non-empty, 0 otherwise
  fin_covenant_count = count of non-missing financial covenant columns (13 structured fields)
                       plus 1 if an EBITDA-level covenant exists (ebitda_initial or ebitda_final non-missing)
  gen_covenant_count = count of general covenant indicators:
                       material_restriction == 'Yes' plus non-missing sweep/release columns (6 fields)
  is_covenant_ratio  = income-statement covenants / fin_covenant_count; missing if fin_covenant_count == 0
  secured            = 1 if secured == 'Yes', 0 if 'No'; overwrites raw Yes/No string
"""

from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import requests

REPO_DIR      = Path(__file__).resolve().parent
DATA_DIR      = REPO_DIR.parent / "data"
RAW_FILE      = DATA_DIR / "dealscan_raw.parquet"
OUT_FILE      = DATA_DIR / "dealscan.parquet"
VAR_LIST_FILE = REPO_DIR.parent / "documentation" / "variables_dealscan.txt"
BOX_URL       = "https://ucdavis.box.com/shared/static/pkcfg63i84lqfza5qm1jlnne8yzmcqvy.csv"

CONSTRUCTED_COLS = [
    "maturity", "log_lender_count", "log_interest", "log_deal_amount",
    "perf_pricing", "fin_covenant_count", "gen_covenant_count", "is_covenant_ratio",
    "secured",
]


def download_raw() -> pd.DataFrame:
    """Download CSV from Box, save as dealscan_raw.parquet, return DataFrame."""
    tmp = DATA_DIR / "Dealscan_1981_2025_full.csv"
    max_retries = 5

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading Dealscan_1981_2025_full.csv from Box (attempt {attempt}/{max_retries})...")
            with requests.get(BOX_URL, stream=True, allow_redirects=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                downloaded = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"  {downloaded / 1e6:.0f} MB / {total / 1e6:.0f} MB ({pct:.1f}%)", end="\r")
            print("\nDownload complete. Reading CSV...")
            break
        except Exception as e:
            print(f"\n  Download interrupted: {e}")
            if tmp.exists():
                tmp.unlink()
            if attempt == max_retries:
                raise RuntimeError(f"Download failed after {max_retries} attempts.") from e
            print("  Retrying...")

    df = pd.read_csv(tmp, low_memory=False)
    tmp.unlink()
    df.to_parquet(RAW_FILE, index=False)
    print(f"Raw data saved to {RAW_FILE}")
    return df


FIN_COVENANT_COLS = [
    "max_leverage_ratio",
    "max_debt_to_cash_flow",
    "max_sr_debt_to_cash_flow",
    "tangible_net_worth",
    "net_worth",
    "min_fixed_charge_coverage_ratio",
    "min_debt_service_coverage_ratio",
    "min_interest_coverage_ratio",
    "min_cash_interest_coverage_ratio",
    "max_debt_to_tangible_net_worth",
    "max_debt_to_equity_ratio",
    "min_current_ratio",
    "max_loan_to_value_ratio",
]

# Income-statement-based subset of FIN_COVENANT_COLS (excludes balance sheet covenants)
IS_COVENANT_COLS = [
    "max_debt_to_cash_flow",
    "max_sr_debt_to_cash_flow",
    "min_fixed_charge_coverage_ratio",
    "min_debt_service_coverage_ratio",
    "min_interest_coverage_ratio",
    "min_cash_interest_coverage_ratio",
]

# material_restriction is a universal Yes/No field — counted via equality check, not notna()
GEN_COVENANT_NUMERIC_COLS = [
    "excess_cf_sweep",
    "asset_sales_sweep",
    "debt_issue_sweep",
    "equity_issue_sweep",
    "insurance_proceeds_sweep",
    "collateral_release",
]


def construct_variables(df: pd.DataFrame) -> pd.DataFrame:
    df["tranche_active_date"]   = pd.to_datetime(df["tranche_active_date"],   errors="coerce")
    df["tranche_maturity_date"] = pd.to_datetime(df["tranche_maturity_date"], errors="coerce")

    days = (df["tranche_maturity_date"] - df["tranche_active_date"]).dt.days
    months = days / (365.25 / 12)
    df["maturity"] = np.where(months > 0, np.log(months), np.nan)

    lenders = df["number_of_lenders"].where(df["number_of_lenders"] > 0)
    df["log_lender_count"] = np.log(lenders)

    spread = df["all_in_spread_drawn_bps"].where(df["all_in_spread_drawn_bps"] > 0)
    df["log_interest"] = np.log(spread)

    deal_amt = df["deal_amount_converted"].where(df["deal_amount_converted"] > 0)
    df["log_deal_amount"] = np.log(deal_amt)

    df["perf_pricing"] = (df["performance_pricing"].notna() & (df["performance_pricing"].str.strip() != "")).astype(int)

    ebitda_cov = (df["ebitda_initial_amount_usd"].notna() | df["ebitda_final_amount_usd"].notna()).astype(int)
    df["fin_covenant_count"] = (df[FIN_COVENANT_COLS].notna().sum(axis=1) + ebitda_cov).astype("Int64")

    is_count = df[IS_COVENANT_COLS].notna().sum(axis=1) + ebitda_cov
    df["is_covenant_ratio"] = (is_count / df["fin_covenant_count"]).where(df["fin_covenant_count"] > 0)

    mat_restrict = (df["material_restriction"] == "Yes").astype(int)
    gen_numeric  = df[GEN_COVENANT_NUMERIC_COLS].notna().sum(axis=1)
    df["gen_covenant_count"] = (mat_restrict + gen_numeric).astype("Int64")

    df["secured"] = (df["secured"].str.strip() == "Yes").astype(int)

    return df


def describe_sample(df: pd.DataFrame) -> None:
    print(f"\nTotal rows: {len(df):,}")

    if "tranche_active_date" in df.columns:
        print(f"\nDate range (tranche_active_date):")
        print(f"  Min: {df['tranche_active_date'].min()}")
        print(f"  Max: {df['tranche_active_date'].max()}")

    print("\nNon-missing counts and basic stats on constructed variables:")
    print(df[["maturity", "log_lender_count", "log_interest", "log_deal_amount", "perf_pricing",
              "fin_covenant_count", "gen_covenant_count", "is_covenant_ratio", "secured"]].describe().round(3).to_string())

    print("\nFirst 5 rows (key columns):")
    cols = ["tranche_active_date", "tranche_maturity_date", "maturity", "number_of_lenders",
            "log_lender_count", "log_interest", "log_deal_amount", "perf_pricing",
            "fin_covenant_count", "gen_covenant_count", "is_covenant_ratio"]
    print(df[[c for c in cols if c in df.columns]].head())


def write_variable_list(df: pd.DataFrame) -> None:
    n_constructed = len(CONSTRUCTED_COLS)
    n_raw         = len(df.columns) - n_constructed
    total         = len(df.columns)

    lines = [
        "DealScan Variable List",
        "Source: Dealscan_1981_2025_full.csv (Dr. Ayung Tseng, UC Davis)",
        f"Total columns: {total} ({n_raw} original + {n_constructed} constructed)",
        f"Generated: {date.today()}",
        "=" * 80,
        "",
        "IDENTIFIERS",
        "-----------",
        "lender_parent_id, deal_active_date, lender_parent_name, lender_name, lender_id,",
        "lpc_deal_id, deal_permid, lpc_tranche_id, tranche_permid, borrower_id,",
        "borrower_name, ticker, perm_id, legal_entity_id_lei",
        "",
        "BORROWER INFO",
        "-------------",
        "city, state_province, zip, country, region, sic_code, naic,",
        "broad_industry_group, major_industry_group, borrower_type, organization_type,",
        "parent, parent_ticker, sponsor, sales_size, sales_size_at_close,",
        "senior_debt_to_ebitda, total_debt_to_ebitda, company_url, guarantor, target,",
        "additional_borrowers",
        "",
        "LENDER ROLES",
        "------------",
        "primary_role, additional_roles, lead_arranger, number_of_lead_arrangers,",
        "bookrunner, number_of_bookrunners, top_tier_arranger, number_of_top_tier_arrangers,",
        "lead_left, number_of_lead_left, arranger, number_of_arrangers,",
        "co_arranger, number_of_co_arrangers, agent, number_of_agents,",
        "lead_manager, number_of_lead_managers, all_lenders, number_of_lenders,",
        "lender_commit, lender_share, lender_region, lender_parent_region,",
        "lender_institution_type, lender_operating_country, lender_parent_operating_country",
        "",
        "DEAL-LEVEL",
        "----------",
        "deal_amount, deal_amount_converted, deal_currency, deal_purpose,",
        "deal_active, deal_refinancing, deal_amended, deal_input_date,",
        "phase, project_finance, purpose_remark, deal_remark, new_money,",
        "new_money_converted, amend_extend_flag, tranche_amended",
        "",
        "TRANCHE-LEVEL",
        "-------------",
        "tranche_active_date, tranche_maturity_date, tranche_amount,",
        "tranche_amount_converted, tranche_currency, tranche_type, seniority_type,",
        "secured, repayment_type, repayment_schedule, market_segment,",
        "market_of_syndication, country_of_syndication, distribution_method,",
        "primary_purpose, secondary_purpose, tertiary_purpose, tenor_maturity,",
        "tranche_o_a, tranche_cusip, tranche_remark, tranche_refinancing,",
        "tranche_active, sponsored, project_finance_sponsor, pro_rata,",
        "closed_date, completion_date, mandated_date, launch_date, average_life,",
        "multi_currency_tranche, league_table_credit, league_table_tranche_date,",
        "league_table_amount, league_table_amount_converted, _100_percent_vote,",
        "currency_onshore_offshore, lin",
        "",
        "PRICING",
        "-------",
        "base_reference_rate, margin_bps, all_base_rate_spread_margin,",
        "base_rate_margin_bps, all_in_spread_drawn_bps, all_in_spread_undrawn_bps,",
        "floor_bps, original_issue_discount, call_protection, call_protection_text,",
        "commitment_fee_bps, upfront_fee_bps, letter_of_credit_fee_bps,",
        "annual_fee_bps, cancellation_fee_bps, utilization_fee_bps,",
        "documentary_issuing_fee_bps, documentary_lc_fee_bps, tiered_upfront_fee,",
        "all_in_fee_asia_only, all_fees, performance_pricing,",
        "performance_pricing_grid, performance_pricing_remark,",
        "assignment_minimum, assignment_fee, base_rate_comment, repayment_comment",
        "",
        "COVENANTS",
        "---------",
        "covenants, max_leverage_ratio, max_debt_to_cash_flow,",
        "max_sr_debt_to_cash_flow, min_interest_coverage_ratio,",
        "min_cash_interest_coverage_ratio, min_fixed_charge_coverage_ratio,",
        "min_debt_service_coverage_ratio, min_current_ratio,",
        "max_debt_to_tangible_net_worth, max_debt_to_equity_ratio,",
        "max_loan_to_value_ratio, tangible_net_worth, net_worth,",
        "all_covenants_financial, covenant_comment,",
        "ebitda_initial_amount_usd, ebitda_final_amount_usd,",
        "capex_initial_usd, capex_final_usd, all_covenants_general,",
        "excess_cf_sweep, asset_sales_sweep, debt_issue_sweep,",
        "equity_issue_sweep, insurance_proceeds_sweep, material_restriction,",
        "percentage_of_net_income, collateral_release, required_lenders,",
        "terms_changes, borrower_consent, agent_consent",
        "",
        "COLLATERAL / SWEEPS",
        "-------------------",
        "collateral_security_type, accounts_receivable, acc_rec_domestic,",
        "acc_rec_foreign, inventory, inv_raw_material, inv_finished_goods,",
        "inv_work_in_progress, cash_cash_equivalents, property_plant_equipment,",
        "marketable_securities, eligible_property_value, oil_gas_reserves",
        "",
        "LOAN FEATURES",
        "-------------",
        "letter_of_credit, swingline, multi_currency, bid_option,",
        "bankers_acceptance, foreign_exchange",
        "",
        "LEGAL",
        "-----",
        "law_firm_name, law_firm_lender_primary, law_firm_lender_other,",
        "law_firm_borrower_primary, law_firm_borrower_other",
        "",
        "CONSTRUCTED VARIABLES",
        "---------------------",
        "maturity          — log(months between tranche_active_date and tranche_maturity_date)",
        "                    months = days / (365.25 / 12); missing if dates missing or diff <= 0",
        "",
        "log_lender_count  — log(number_of_lenders); missing if number_of_lenders <= 0",
        "",
        "log_interest      — log(all_in_spread_drawn_bps); missing if spread <= 0",
        "",
        "log_deal_amount   — log(deal_amount_converted); USD-converted deal size; missing if amount <= 0",
        "",
        "perf_pricing      — 1 if performance_pricing field is non-empty, 0 otherwise",
        "",
        "fin_covenant_count — count of financial covenants present (0–14):",
        "                     13 structured threshold columns (non-missing = present) +",
        "                     1 for EBITDA covenant (ebitda_initial_amount_usd OR ebitda_final_amount_usd non-missing)",
        "                     See documentation/covenant_variable_classification.txt for full list and IS/BS breakdown.",
        "",
        "gen_covenant_count — count of general covenants present (0–7):",
        "                     material_restriction == 'Yes' plus non-missing sweep/release columns",
        "                     See documentation/covenant_variable_classification.txt for full list.",
        "",
        "is_covenant_ratio  — income-statement covenants / fin_covenant_count",
        "                     missing when fin_covenant_count == 0; ranges from 0 to 1",
        "",
        "secured            — 1 if secured == 'Yes', 0 if 'No'; recoded from raw Yes/No string",
    ]

    VAR_LIST_FILE.write_text("\n".join(lines) + "\n")
    print(f"Variable list written to {VAR_LIST_FILE}")


def main() -> None:
    # Step 1: ensure raw parquet exists (download only if missing)
    if RAW_FILE.exists():
        print(f"Loading raw data from {RAW_FILE}")
        df_raw = pd.read_parquet(RAW_FILE)
    else:
        print(f"{RAW_FILE.name} not found — downloading from Box...")
        df_raw = download_raw()

    # Step 2: always rebuild constructed variables from raw
    print("Building constructed variables...")
    df = construct_variables(df_raw.copy())
    df.to_parquet(OUT_FILE, index=False)
    print(f"Analysis file saved to {OUT_FILE}")

    describe_sample(df)
    write_variable_list(df)


if __name__ == "__main__":
    main()
