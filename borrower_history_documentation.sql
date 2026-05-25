/*
Lender Experience / Borrower History Pipeline (PostgreSQL)

Purpose
  Build lender-experience measures for each focal contract (borrower_id, tranche_active_date) by:
    1) defining a focal sample from EDGAR-derived contracts (ListA),
    2) defining a historical DealScan lender-deal universe (ListB),
    3) constructing prior-client histories (24-month lookback) at the lender-parent and lender-branch levels,
    4) joining prior-client histories to Audit Analytics modules (AEC, AQRM, FR, IC, etc.) to compute event counts
       and select a single "winner" lender per borrower–tranche using your tie-break logic,
    5) merging dependent variable contracts with IVs and control variables into a modeling dataset.

Key design choices (matches your decisions)
  - Lookback window: prior events are those with file_date in
        [ tranche_active_date - 24 months , tranche_active_date - 1 day ]
    i.e., inclusive lower bound and inclusive upper bound of "day before tranche".

  - other_count: a lender prominence measure defined as the number of DISTINCT borrowers a lender parent
    finances in the FULL DealScan universe (ListB), regardless of time. This is used only as a scaling
    denominator for ratios.

  - Lead arrangers: you pre-selected a single tranche per contract, so lead-arranger lists are treated as
    deal-level and applied to the selected tranche.

  - Mapping across datasets:
      * DealScan borrower_id == borrower_link_table.dealscan_id
      * borrower_link_table is the canonical bridge between DealScan borrower identifiers and Compustat gvkey
      * DealScan client_cik == Audit Analytics company_fkey (so AA module joins use client_cik directly)

Style conventions
  - SQL keywords are UPPERCASE.
  - Every statement ends with a semicolon.
  - Schema-qualified names are not used.

Run notes
  - This script assumes the following upstream tables already exist:
      contracts, dealscan, borrower_link_table, compustat,
      and Audit Analytics module tables referenced later (e.g., internal_controls).
  - This script also assumes your module-specific lender-experience tables (aec_outside_all, etc.)
    are produced by earlier sections (not shown in full below). The merge step for those tables is included.
*/

BEGIN;

/* -------------------------------------------------------------------------- */
/* 1) ListA: focal lender-deal rows from EDGAR-derived contracts               */
/* -------------------------------------------------------------------------- */
/*
Grain
  One row per (lender_parent_id, lender_id, lpc_deal_id).

Rule
  Keep the earliest tranche_active_date within each lender-deal.
*/

DROP TABLE IF EXISTS list_a;

CREATE TABLE list_a AS
SELECT DISTINCT ON (lender_parent_id, lender_id, lpc_deal_id)
       borrower_id,
       lpc_deal_id,
       tranche_active_date,
       lender_parent_id,
       lender_id
FROM contracts
ORDER BY lpc_deal_id,
         lender_parent_id,
         lender_id,
         tranche_active_date ASC NULLS LAST;

CREATE INDEX IF NOT EXISTS ix_list_a_keys
    ON list_a (lender_parent_id, lender_id, lpc_deal_id, tranche_active_date);


/* -------------------------------------------------------------------------- */
/* 2) ListB: historical lender-deal pool from DealScan                         */
/* -------------------------------------------------------------------------- */
/*
Grain
  One row per (lender_parent_id, lender_id, lpc_deal_id).

Rule
  Keep the earliest tranche_active_date within each lender-deal.
*/

DROP TABLE IF EXISTS list_b;

CREATE TABLE list_b AS
SELECT DISTINCT ON (lender_parent_id, lender_id, lpc_deal_id)
       borrower_id,
       lpc_deal_id,
       tranche_active_date,
       lender_parent_id,
       lender_id,
       lender_parent_name,
       lender_name
FROM dealscan
ORDER BY lpc_deal_id,
         lender_parent_id,
         lender_id,
         tranche_active_date ASC NULLS LAST;

CREATE INDEX IF NOT EXISTS ix_list_b_keys
    ON list_b (lender_parent_id, lender_id, tranche_active_date, borrower_id);


/* -------------------------------------------------------------------------- */
/* 3) other_count: lender prominence denominator (GLOBAL, time-invariant)     */
/* -------------------------------------------------------------------------- */
/*
Definition
  other_count = number of DISTINCT borrowers a lender parent finances in the FULL ListB universe.

Interpretation
  Larger other_count means the lender parent is active with more distinct borrowers in DealScan.
  Ratios such as (event_count / other_count) scale event counts by lender prominence.
*/

DROP TABLE IF EXISTS lender_parent_other_counts;

CREATE TABLE lender_parent_other_counts AS
SELECT
    lender_parent_id,
    MAX(lender_parent_name) AS lender_parent_name,
    COUNT(DISTINCT borrower_id) AS other_count
FROM list_b
WHERE lender_parent_id IS NOT NULL
GROUP BY lender_parent_id;

CREATE INDEX IF NOT EXISTS ix_lender_parent_other_counts
    ON lender_parent_other_counts (lender_parent_id);


/* -------------------------------------------------------------------------- */
/* 4) Prior-client tables: 24-month lookback relative to ListA                 */
/* -------------------------------------------------------------------------- */
/*
Lookback window
  Prior relationship tranche dates are in:
      [ a.tranche_active_date - 24 months , a.tranche_active_date - 1 day ]

Self indicator
  self = 1 if the prior borrower matches the focal borrower.

Note
  These tables are the "spine" for joining to Audit Analytics modules.
*/

DROP TABLE IF EXISTS parent_clients;

CREATE TABLE parent_clients AS
SELECT
    a.borrower_id,
    a.lpc_deal_id,
    a.tranche_active_date,
    a.lender_parent_id,
    oc.lender_parent_name,
    a.lender_id,
    b.borrower_id AS previous_parent_client_id,
    CAST(CASE WHEN b.borrower_id = a.borrower_id THEN 1 ELSE 0 END AS INT) AS self,
    oc.other_count
FROM list_a a
LEFT JOIN list_b b
    ON b.lender_parent_id = a.lender_parent_id
   AND b.tranche_active_date BETWEEN (a.tranche_active_date - INTERVAL '24 months')
                                AND (a.tranche_active_date - INTERVAL '1 day')
LEFT JOIN lender_parent_other_counts oc
    ON oc.lender_parent_id = a.lender_parent_id;

CREATE INDEX IF NOT EXISTS ix_parent_clients_keys
    ON parent_clients (lender_parent_id, tranche_active_date, borrower_id);


DROP TABLE IF EXISTS branch_clients;

CREATE TABLE branch_clients AS
SELECT
    a.borrower_id,
    a.lpc_deal_id,
    a.tranche_active_date,
    a.lender_parent_id,
    a.lender_id,
    b.borrower_id AS previous_branch_client_id,
    CAST(CASE WHEN b.borrower_id = a.borrower_id THEN 1 ELSE 0 END AS INT) AS self
FROM list_a a
LEFT JOIN list_b b
    ON b.lender_parent_id = a.lender_parent_id
   AND b.lender_id = a.lender_id
   AND b.tranche_active_date BETWEEN (a.tranche_active_date - INTERVAL '24 months')
                                AND (a.tranche_active_date - INTERVAL '1 day');

CREATE INDEX IF NOT EXISTS ix_branch_clients_keys
    ON branch_clients (lender_parent_id, lender_id, tranche_active_date, borrower_id);


/* -------------------------------------------------------------------------- */
/* 5) Borrower ↔ CIK/GVKEY mapping (cleaned)                                   */
/* -------------------------------------------------------------------------- */
/*
Goal
  Build a clean mapping from DealScan borrower_id to CIK/GVKEY.

Why this is needed
  In your data, a single (CIK, GVKEY) may map to one OR two borrower_ids.
  Instead of maintaining two separate update passes, we normalize to a long mapping table
  with one row per borrower_id.

Assumptions
  - borrower_link_table.dealscan_id equals dealscan.borrower_id.
  - borrower_link_table.dealscan_id2 is an optional second borrower_id for the same firm.
  - compustat contains the authoritative cik for each gvkey.
*/

DROP TABLE IF EXISTS borrower_cik;

CREATE TABLE borrower_cik AS
WITH link_long AS (
    SELECT
        blt.gvkey,
        blt.dealscan_id AS borrower_id
    FROM borrower_link_table blt
    WHERE blt.dealscan_id IS NOT NULL

    UNION ALL

    SELECT
        blt.gvkey,
        blt.dealscan_id2 AS borrower_id
    FROM borrower_link_table blt
    WHERE blt.dealscan_id2 IS NOT NULL
),
link_dedup AS (
    SELECT DISTINCT gvkey, borrower_id
    FROM link_long
    WHERE borrower_id IS NOT NULL
)
SELECT
    ld.borrower_id,
    c.cik,
    ld.gvkey
FROM link_dedup ld
LEFT JOIN compustat c
    ON c.gvkey = ld.gvkey;

CREATE INDEX IF NOT EXISTS ix_borrower_cik_borrower
    ON borrower_cik (borrower_id);


/* -------------------------------------------------------------------------- */
/* 6) Example module table: Internal Controls (inside industry, all lenders)   */
/* -------------------------------------------------------------------------- */
/*
This block matches your module pattern:
  1) join prior clients (parent_clients) to AA module rows (internal_controls) using client_cik/company_fkey,
  2) restrict to the 24-month lookback window ending at tranche_active_date - 1 day,
  3) compute per-lender counts and ratios (scaled by other_count),
  4) within each borrower–tranche, compute tranche-level "all-zero" flag,
  5) pick a single winner lender using your tie-break rule, with majority lead-arranger fallback.

Notes
  - The inside/outside industry split is handled in earlier upstream steps (not repeated here).
  - op_type codes are assumed to be those used in your data (e.g., 'a' and 'm').
*/

DROP TABLE IF EXISTS ic_inside_all;

CREATE TABLE ic_inside_all AS
WITH joined AS (
    SELECT
        p.lender_parent_id,
        p.lender_parent_name,
        p.borrower_id,
        p.tranche_active_date,
        p.other_count,
        (p.lender_parent_id = ANY (COALESCE(p.arrangers_id_list, ARRAY[]::INT[]))) AS is_lead,
        p.grouping,
        i.ic_is_effective,
        i.ic_op_type
    FROM parent_clients p
    LEFT JOIN internal_controls i
        ON i.company_fkey = p.client_cik
       AND i.file_date   BETWEEN (p.tranche_active_date - INTERVAL '24 months')
                            AND (p.tranche_active_date - INTERVAL '1 day')
),
norm AS (
    SELECT
        lender_parent_id,
        lender_parent_name,
        borrower_id,
        tranche_active_date,
        other_count,
        is_lead,
        grouping,
        CASE
            WHEN UPPER(TRIM(COALESCE(ic_is_effective, ''))) IN ('N','0','NO','FALSE','F') THEN TRUE
            ELSE FALSE
        END AS is_ineffective,
        LOWER(TRIM(ic_op_type)) AS op_type
    FROM joined
),
per_lender AS (
    SELECT
        lender_parent_id,
        lender_parent_name,
        borrower_id,
        tranche_active_date,
        is_lead,
        grouping,
        other_count,
        COUNT(*) FILTER (WHERE is_ineffective AND op_type = 'a') AS ic_sum_a_ineff,
        COUNT(*) FILTER (WHERE is_ineffective AND op_type = 'm') AS ic_sum_m_ineff
    FROM norm
    GROUP BY lender_parent_id, lender_parent_name, borrower_id, tranche_active_date, is_lead, grouping, other_count
),
with_ratios AS (
    SELECT
        pl.*,
        pl.ic_sum_a_ineff::FLOAT / NULLIF(pl.other_count, 0) AS ic_ratio_a_ineff,
        pl.ic_sum_m_ineff::FLOAT / NULLIF(pl.other_count, 0) AS ic_ratio_m_ineff
    FROM per_lender pl
),
all_zero_flag AS (
    SELECT
        borrower_id,
        tranche_active_date,
        BOOL_AND(ic_sum_a_ineff = 0 AND ic_sum_m_ineff = 0) AS all_sums_zero
    FROM with_ratios
    GROUP BY borrower_id, tranche_active_date
),
lead_majority_group AS (
    SELECT
        borrower_id,
        tranche_active_date,
        grouping,
        ROW_NUMBER() OVER (
            PARTITION BY borrower_id, tranche_active_date
            ORDER BY COUNT(*) DESC, grouping ASC NULLS LAST
        ) AS rk
    FROM with_ratios
    WHERE is_lead AND grouping IS NOT NULL
    GROUP BY borrower_id, tranche_active_date, grouping
),
lead_majority_group_top AS (
    SELECT borrower_id, tranche_active_date, grouping AS majority_grouping
    FROM lead_majority_group
    WHERE rk = 1
),
rolled AS (
    SELECT
        borrower_id,
        tranche_active_date,
        MAX(ic_sum_a_ineff)   AS ic_sum_a_ineff_max,
        MAX(ic_sum_m_ineff)   AS ic_sum_m_ineff_max,
        MAX(ic_ratio_a_ineff) AS ic_ratio_a_ineff_max,
        MAX(ic_ratio_m_ineff) AS ic_ratio_m_ineff_max
    FROM with_ratios
    GROUP BY borrower_id, tranche_active_date
),
pick_best AS (
    SELECT
        l.*,
        ROW_NUMBER() OVER (
            PARTITION BY l.borrower_id, l.tranche_active_date
            ORDER BY
                CASE
                    WHEN z.all_sums_zero
                     AND (l.grouping IS NOT DISTINCT FROM mg.majority_grouping) THEN 0
                    WHEN z.all_sums_zero THEN 1
                    ELSE 2
                END,
                l.ic_sum_a_ineff   DESC,
                l.ic_sum_m_ineff   DESC,
                l.ic_ratio_a_ineff DESC NULLS LAST,
                l.ic_ratio_m_ineff DESC NULLS LAST,
                l.grouping         ASC NULLS LAST,
                l.lender_parent_id ASC
        ) AS rn
    FROM with_ratios l
    LEFT JOIN all_zero_flag z
        ON z.borrower_id = l.borrower_id
       AND z.tranche_active_date = l.tranche_active_date
    LEFT JOIN lead_majority_group_top mg
        ON mg.borrower_id = l.borrower_id
       AND mg.tranche_active_date = l.tranche_active_date
),
best_one AS (
    SELECT
        borrower_id,
        tranche_active_date,
        lender_parent_id   AS selected_parent_lender_id,
        lender_parent_name AS selected_parent_lender_name,
        grouping           AS selected_grouping
    FROM pick_best
    WHERE rn = 1
)
SELECT
    r.borrower_id,
    r.tranche_active_date,
    r.ic_sum_a_ineff_max,
    r.ic_sum_m_ineff_max,
    r.ic_ratio_a_ineff_max,
    r.ic_ratio_m_ineff_max,
    CASE
        WHEN z.all_sums_zero THEN mg.majority_grouping
        ELSE b.selected_grouping
    END AS lead_group_largest,
    b.selected_parent_lender_id,
    b.selected_parent_lender_name,
    COALESCE(z.all_sums_zero, FALSE) AS used_majority_fallback
FROM rolled r
LEFT JOIN best_one b
    ON b.borrower_id = r.borrower_id
   AND b.tranche_active_date = r.tranche_active_date
LEFT JOIN all_zero_flag z
    ON z.borrower_id = r.borrower_id
   AND z.tranche_active_date = r.tranche_active_date
LEFT JOIN lead_majority_group_top mg
    ON mg.borrower_id = r.borrower_id
   AND mg.tranche_active_date = r.tranche_active_date;


/* -------------------------------------------------------------------------- */
/* 7) Merge AA module outputs into a single borrower–tranche table             */
/* -------------------------------------------------------------------------- */
/*
This merge assumes that the following module output tables already exist:
  aec_outside_all, aec_outside_lead, aec_inside_all, aec_inside_lead,
  aqrm_outside_all, aqrm_outside_lead, aqrm_inside_all, aqrm_inside_lead,
  fr_outside_all, fr_outside_lead, fr_inside_all, fr_inside_lead,
  ic_outside_all, ic_outside_lead, ic_inside_all, ic_inside_lead.

All joins are on (borrower_id, tranche_active_date).
*/

DROP TABLE IF EXISTS full_parent_aa_event_counts;

CREATE TABLE full_parent_aa_event_counts AS
SELECT
    aec1.borrower_id,
    aec1.tranche_active_date,

    /* --- AEC (Accounting & Auditing Enforcement / Estimates) --- */
    aec1.selected_parent_lender_id AS selected_parent_lender_id_aec_oa,
    aec1.lead_group_largest        AS selected_grouping_aec_oa,
    aec1.est_pc_sum_max            AS est_pc_sum_max_aec_oa,
    aec1.est_nc_sum_max            AS est_nc_sum_max_aec_oa,

    aec2.selected_parent_lender_id AS selected_parent_lender_id_aec_ol,
    aec2.lead_group_largest        AS selected_grouping_aec_ol,
    aec2.est_pc_sum_max            AS est_pc_sum_max_aec_ol,
    aec2.est_nc_sum_max            AS est_nc_sum_max_aec_ol,

    aec3.selected_parent_lender_id AS selected_parent_lender_id_aec_ia,
    aec3.lead_group_largest        AS selected_grouping_aec_ia,
    aec3.est_pc_sum_max            AS est_pc_sum_max_aec_ia,
    aec3.est_nc_sum_max            AS est_nc_sum_max_aec_ia,

    aec4.selected_parent_lender_id AS selected_parent_lender_id_aec_il,
    aec4.lead_group_largest        AS selected_grouping_aec_il,
    aec4.est_pc_sum_max            AS est_pc_sum_max_aec_il,
    aec4.est_nc_sum_max            AS est_nc_sum_max_aec_il,

    /* --- AQRM --- */
    aqrm1.selected_parent_lender_id AS selected_parent_lender_id_aqrm_oa,
    aqrm1.lead_group_largest        AS selected_grouping_aqrm_oa,
    aqrm1.aqrm_gc_sum_max           AS aqrm_gc_sum_max_aqrm_oa,
    aqrm1.aqrm_ceo_sum_max          AS aqrm_ceo_sum_max_aqrm_oa,
    aqrm1.aqrm_lf_sum_max           AS aqrm_lf_sum_max_aqrm_oa,
    aqrm1.aqrm_mi_sum_max           AS aqrm_mi_sum_max_aqrm_oa,

    aqrm2.selected_parent_lender_id AS selected_parent_lender_id_aqrm_ol,
    aqrm2.lead_group_largest        AS selected_grouping_aqrm_ol,
    aqrm2.aqrm_gc_sum_max           AS aqrm_gc_sum_max_aqrm_ol,
    aqrm2.aqrm_ceo_sum_max          AS aqrm_ceo_sum_max_aqrm_ol,
    aqrm2.aqrm_lf_sum_max           AS aqrm_lf_sum_max_aqrm_ol,
    aqrm2.aqrm_mi_sum_max           AS aqrm_mi_sum_max_aqrm_ol,

    aqrm3.selected_parent_lender_id AS selected_parent_lender_id_aqrm_ia,
    aqrm3.lead_group_largest        AS selected_grouping_aqrm_ia,
    aqrm3.aqrm_gc_sum_max           AS aqrm_gc_sum_max_aqrm_ia,
    aqrm3.aqrm_ceo_sum_max          AS aqrm_ceo_sum_max_aqrm_ia,
    aqrm3.aqrm_lf_sum_max           AS aqrm_lf_sum_max_aqrm_ia,
    aqrm3.aqrm_mi_sum_max           AS aqrm_mi_sum_max_aqrm_ia,

    aqrm4.selected_parent_lender_id AS selected_parent_lender_id_aqrm_il,
    aqrm4.lead_group_largest        AS selected_grouping_aqrm_il,
    aqrm4.aqrm_gc_sum_max           AS aqrm_gc_sum_max_aqrm_il,
    aqrm4.aqrm_ceo_sum_max          AS aqrm_ceo_sum_max_aqrm_il,
    aqrm4.aqrm_lf_sum_max           AS aqrm_lf_sum_max_aqrm_il,
    aqrm4.aqrm_mi_sum_max           AS aqrm_mi_sum_max_aqrm_il,

    /* --- FR (Financial Restatements) --- */
    fr1.selected_parent_lender_id AS selected_parent_lender_id_fr_oa,
    fr1.lead_group_largest        AS selected_grouping_fr_oa,
    fr1.res_sum_adv1_max          AS res_sum_adv1_max_fr_oa,
    fr1.res_sum_adv0_max          AS res_sum_adv0_max_fr_oa,
    fr1.res_fraud_sum_adv1_max    AS res_fraud_sum_adv1_max_fr_oa,
    fr1.res_fraud_sum_adv0_max    AS res_fraud_sum_adv0_max_fr_oa,
    fr1.res_cler_sum_adv1_max     AS res_cler_sum_adv1_max_fr_oa,
    fr1.res_cler_sum_adv0_max     AS res_cler_sum_adv0_max_fr_oa,
    fr1.lease_res_sum_adv1_max    AS lease_res_sum_adv1_max_fr_oa,
    fr1.lease_res_sum_adv0_max    AS lease_res_sum_adv0_max_fr_oa,
    fr1.lease_res_fraud_sum_adv1_max AS lease_res_fraud_sum_adv1_max_fr_oa,
    fr1.lease_res_fraud_sum_adv0_max AS lease_res_fraud_sum_adv0_max_fr_oa,
    fr1.lease_res_cler_sum_adv1_max  AS lease_res_cler_sum_adv1_max_fr_oa,
    fr1.lease_res_cler_sum_adv0_max  AS lease_res_cler_sum_adv0_max_fr_oa,

    fr2.selected_parent_lender_id AS selected_parent_lender_id_fr_ol,
    fr2.lead_group_largest        AS selected_grouping_fr_ol,
    fr2.res_sum_adv1_max          AS res_sum_adv1_max_fr_ol,
    fr2.res_sum_adv0_max          AS res_sum_adv0_max_fr_ol,
    fr2.res_fraud_sum_adv1_max    AS res_fraud_sum_adv1_max_fr_ol,
    fr2.res_fraud_sum_adv0_max    AS res_fraud_sum_adv0_max_fr_ol,
    fr2.res_cler_sum_adv1_max     AS res_cler_sum_adv1_max_fr_ol,
    fr2.res_cler_sum_adv0_max     AS res_cler_sum_adv0_max_fr_ol,
    fr2.lease_res_sum_adv1_max    AS lease_res_sum_adv1_max_fr_ol,
    fr2.lease_res_sum_adv0_max    AS lease_res_sum_adv0_max_fr_ol,
    fr2.lease_res_fraud_sum_adv1_max AS lease_res_fraud_sum_adv1_max_fr_ol,
    fr2.lease_res_fraud_sum_adv0_max AS lease_res_fraud_sum_adv0_max_fr_ol,
    fr2.lease_res_cler_sum_adv1_max  AS lease_res_cler_sum_adv1_max_fr_ol,
    fr2.lease_res_cler_sum_adv0_max  AS lease_res_cler_sum_adv0_max_fr_ol,

    fr3.selected_parent_lender_id AS selected_parent_lender_id_fr_ia,
    fr3.lead_group_largest        AS selected_grouping_fr_ia,
    fr3.res_sum_adv1_max          AS res_sum_adv1_max_fr_ia,
    fr3.res_sum_adv0_max          AS res_sum_adv0_max_fr_ia,
    fr3.res_fraud_sum_adv1_max    AS res_fraud_sum_adv1_max_fr_ia,
    fr3.res_fraud_sum_adv0_max    AS res_fraud_sum_adv0_max_fr_ia,
    fr3.res_cler_sum_adv1_max     AS res_cler_sum_adv1_max_fr_ia,
    fr3.res_cler_sum_adv0_max     AS res_cler_sum_adv0_max_fr_ia,
    fr3.lease_res_sum_adv1_max    AS lease_res_sum_adv1_max_fr_ia,
    fr3.lease_res_sum_adv0_max    AS lease_res_sum_adv0_max_fr_ia,
    fr3.lease_res_fraud_sum_adv1_max AS lease_res_fraud_sum_adv1_max_fr_ia,
    fr3.lease_res_fraud_sum_adv0_max AS lease_res_fraud_sum_adv0_max_fr_ia,
    fr3.lease_res_cler_sum_adv1_max  AS lease_res_cler_sum_adv1_max_fr_ia,
    fr3.lease_res_cler_sum_adv0_max  AS lease_res_cler_sum_adv0_max_fr_ia,

    fr4.selected_parent_lender_id AS selected_parent_lender_id_fr_il,
    fr4.lead_group_largest        AS selected_grouping_fr_il,
    fr4.res_sum_adv1_max          AS res_sum_adv1_max_fr_il,
    fr4.res_sum_adv0_max          AS res_sum_adv0_max_fr_il,
    fr4.res_fraud_sum_adv1_max    AS res_fraud_sum_adv1_max_fr_il,
    fr4.res_fraud_sum_adv0_max    AS res_fraud_sum_adv0_max_fr_il,
    fr4.res_cler_sum_adv1_max     AS res_cler_sum_adv1_max_fr_il,
    fr4.res_cler_sum_adv0_max     AS res_cler_sum_adv0_max_fr_il,
    fr4.lease_res_sum_adv1_max    AS lease_res_sum_adv1_max_fr_il,
    fr4.lease_res_sum_adv0_max    AS lease_res_sum_adv0_max_fr_il,
    fr4.lease_res_fraud_sum_adv1_max AS lease_res_fraud_sum_adv1_max_fr_il,
    fr4.lease_res_fraud_sum_adv0_max AS lease_res_fraud_sum_adv0_max_fr_il,
    fr4.lease_res_cler_sum_adv1_max  AS lease_res_cler_sum_adv1_max_fr_il,
    fr4.lease_res_cler_sum_adv0_max  AS lease_res_cler_sum_adv0_max_fr_il,

    /* --- IC (Internal Controls) --- */
    ic1.selected_parent_lender_id AS selected_parent_lender_id_ic_oa,
    ic1.lead_group_largest        AS selected_grouping_ic_oa,
    ic1.ic_sum_a_ineff_max        AS ic_sum_a_ineff_max_ic_oa,
    ic1.ic_sum_m_ineff_max        AS ic_sum_m_ineff_max_ic_oa,

    ic2.selected_parent_lender_id AS selected_parent_lender_id_ic_ol,
    ic2.lead_group_largest        AS selected_grouping_ic_ol,
    ic2.ic_sum_a_ineff_max        AS ic_sum_a_ineff_max_ic_ol,
    ic2.ic_sum_m_ineff_max        AS ic_sum_m_ineff_max_ic_ol,

    ic3.selected_parent_lender_id AS selected_parent_lender_id_ic_ia,
    ic3.lead_group_largest        AS selected_grouping_ic_ia,
    ic3.ic_sum_a_ineff_max        AS ic_sum_a_ineff_max_ic_ia,
    ic3.ic_sum_m_ineff_max        AS ic_sum_m_ineff_max_ic_ia,

    ic4.selected_parent_lender_id AS selected_parent_lender_id_ic_il,
    ic4.lead_group_largest        AS selected_grouping_ic_il,
    ic4.ic_sum_a_ineff_max        AS ic_sum_a_ineff_max_ic_il,
    ic4.ic_sum_m_ineff_max        AS ic_sum_m_ineff_max_ic_il
FROM aec_outside_all aec1
LEFT JOIN aec_outside_lead aec2
    ON aec1.borrower_id = aec2.borrower_id
   AND aec1.tranche_active_date = aec2.tranche_active_date
LEFT JOIN aec_inside_all aec3
    ON aec1.borrower_id = aec3.borrower_id
   AND aec1.tranche_active_date = aec3.tranche_active_date
LEFT JOIN aec_inside_lead aec4
    ON aec1.borrower_id = aec4.borrower_id
   AND aec1.tranche_active_date = aec4.tranche_active_date
LEFT JOIN aqrm_outside_all aqrm1
    ON aec1.borrower_id = aqrm1.borrower_id
   AND aec1.tranche_active_date = aqrm1.tranche_active_date
LEFT JOIN aqrm_outside_lead aqrm2
    ON aec1.borrower_id = aqrm2.borrower_id
   AND aec1.tranche_active_date = aqrm2.tranche_active_date
LEFT JOIN aqrm_inside_all aqrm3
    ON aec1.borrower_id = aqrm3.borrower_id
   AND aec1.tranche_active_date = aqrm3.tranche_active_date
LEFT JOIN aqrm_inside_lead aqrm4
    ON aec1.borrower_id = aqrm4.borrower_id
   AND aec1.tranche_active_date = aqrm4.tranche_active_date
LEFT JOIN fr_outside_all fr1
    ON aec1.borrower_id = fr1.borrower_id
   AND aec1.tranche_active_date = fr1.tranche_active_date
LEFT JOIN fr_outside_lead fr2
    ON aec1.borrower_id = fr2.borrower_id
   AND aec1.tranche_active_date = fr2.tranche_active_date
LEFT JOIN fr_inside_all fr3
    ON aec1.borrower_id = fr3.borrower_id
   AND aec1.tranche_active_date = fr3.tranche_active_date
LEFT JOIN fr_inside_lead fr4
    ON aec1.borrower_id = fr4.borrower_id
   AND aec1.tranche_active_date = fr4.tranche_active_date
LEFT JOIN ic_outside_all ic1
    ON aec1.borrower_id = ic1.borrower_id
   AND aec1.tranche_active_date = ic1.tranche_active_date
LEFT JOIN ic_outside_lead ic2
    ON aec1.borrower_id = ic2.borrower_id
   AND aec1.tranche_active_date = ic2.tranche_active_date
LEFT JOIN ic_inside_all ic3
    ON aec1.borrower_id = ic3.borrower_id
   AND aec1.tranche_active_date = ic3.tranche_active_date
LEFT JOIN ic_inside_lead ic4
    ON aec1.borrower_id = ic4.borrower_id
   AND aec1.tranche_active_date = ic4.tranche_active_date;


/* Attach CIK to full_parent_aa_event_counts via borrower_id (clean, single pass) */

ALTER TABLE full_parent_aa_event_counts
    ADD COLUMN cik2 INT4;

UPDATE full_parent_aa_event_counts f
SET cik2 = bc.cik
FROM borrower_cik bc
WHERE f.borrower_id = bc.borrower_id
  AND f.cik2 IS NULL;


/* -------------------------------------------------------------------------- */
/* 8) Regression spine: DV + IV merge (DV tables are assumed pre-processed)    */
/* -------------------------------------------------------------------------- */
/*
Per your request:
  - The DV table stack / cleaning across years is removed from this documented script.
  - Provide dv_contracts_clean as an upstream input (post-processed).

Expected dv_contracts_clean columns used downstream
  cik, gvkey, file_link, contract_date, dv
*/

DROP TABLE IF EXISTS full_parent_regression;

CREATE TABLE full_parent_regression AS
SELECT
    dc.*,
    f.*
FROM dv_contracts_clean dc
LEFT JOIN full_parent_aa_event_counts f
    ON dc.cik = f.cik2
   AND dc.contract_date = f.tranche_active_date;

DROP TABLE IF EXISTS full_parent_regression_matches;

CREATE TABLE full_parent_regression_matches AS
SELECT
    dc.*,
    f.*
FROM dv_contracts_clean dc
INNER JOIN full_parent_aa_event_counts f
    ON dc.cik = f.cik2
   AND dc.contract_date = f.tranche_active_date;


/* -------------------------------------------------------------------------- */
/* 9) Final selected IV columns for modeling                                   */
/* -------------------------------------------------------------------------- */

DROP TABLE IF EXISTS full_parent_regression_selected;

CREATE TABLE full_parent_regression_selected AS
SELECT
    fprm.cik,
    fprm.gvkey,
    fprm.file_link,
    fprm.contract_date,
    EXTRACT(YEAR FROM fprm.contract_date) AS contract_year,
    fprm.dv,
    fprm.borrower_id,

    /* --- AEC NC --- */
    selected_parent_lender_id_aec_oa,
    selected_grouping_aec_oa,
    est_nc_sum_max_aec_oa,
    selected_parent_lender_id_aec_ol,
    selected_grouping_aec_ol,
    est_nc_sum_max_aec_ol,
    selected_parent_lender_id_aec_ia,
    selected_grouping_aec_ia,
    est_nc_sum_max_aec_ia,
    selected_parent_lender_id_aec_il,
    selected_grouping_aec_il,
    est_nc_sum_max_aec_il,

    /* --- FR ADV1 --- */
    selected_parent_lender_id_fr_oa,
    selected_grouping_fr_oa,
    res_sum_adv1_max_fr_oa,
    selected_parent_lender_id_fr_ol,
    selected_grouping_fr_ol,
    res_sum_adv1_max_fr_ol,
    selected_parent_lender_id_fr_ia,
    selected_grouping_fr_ia,
    res_sum_adv1_max_fr_ia,
    selected_parent_lender_id_fr_il,
    selected_grouping_fr_il,
    res_sum_adv1_max_fr_il,

    /* --- IC A + M --- */
    selected_parent_lender_id_ic_oa,
    selected_grouping_ic_oa,
    ic_sum_a_ineff_max_ic_oa,
    ic_sum_m_ineff_max_ic_oa,
    selected_parent_lender_id_ic_ol,
    selected_grouping_ic_ol,
    ic_sum_a_ineff_max_ic_ol,
    ic_sum_m_ineff_max_ic_ol,
    selected_parent_lender_id_ic_ia,
    selected_grouping_ic_ia,
    ic_sum_a_ineff_max_ic_ia,
    ic_sum_m_ineff_max_ic_ia,
    selected_parent_lender_id_ic_il,
    selected_grouping_ic_il,
    ic_sum_a_ineff_max_ic_il,
    ic_sum_m_ineff_max_ic_il,

    /* --- AQRM --- */
    selected_parent_lender_id_aqrm_oa,
    selected_grouping_aqrm_oa,
    aqrm_gc_sum_max_aqrm_oa,
    aqrm_lf_sum_max_aqrm_oa,
    aqrm_mi_sum_max_aqrm_oa,
    selected_parent_lender_id_aqrm_ol,
    selected_grouping_aqrm_ol,
    aqrm_gc_sum_max_aqrm_ol,
    aqrm_lf_sum_max_aqrm_ol,
    aqrm_mi_sum_max_aqrm_ol,
    selected_parent_lender_id_aqrm_ia,
    selected_grouping_aqrm_ia,
    aqrm_gc_sum_max_aqrm_ia,
    aqrm_lf_sum_max_aqrm_ia,
    aqrm_mi_sum_max_aqrm_ia,
    selected_parent_lender_id_aqrm_il,
    selected_grouping_aqrm_il,
    aqrm_gc_sum_max_aqrm_il,
    aqrm_lf_sum_max_aqrm_il,
    aqrm_mi_sum_max_aqrm_il
FROM full_parent_regression_matches fprm;


/* -------------------------------------------------------------------------- */
/* 10) Attach contract-level metadata from ListA (industry codes and lenders)  */
/* -------------------------------------------------------------------------- */

ALTER TABLE full_parent_regression_selected
    ADD COLUMN naics VARCHAR(32),
    ADD COLUMN sic VARCHAR(32),
    ADD COLUMN lead_arrangers_count INT4,
    ADD COLUMN lead_arrangers_id_list INT[],
    ADD COLUMN all_lenders_count INT4,
    ADD COLUMN all_lenders_id_list INT[];

UPDATE full_parent_regression_selected fprs
SET naics = l.naics,
    sic = l.sic
FROM list_a l
WHERE l.borrower_id = fprs.borrower_id
  AND l.tranche_active_date = fprs.contract_date;

UPDATE full_parent_regression_selected fprs
SET lead_arrangers_id_list = l.lead_arrangers_id_list
FROM list_a l
WHERE l.borrower_id = fprs.borrower_id
  AND l.tranche_active_date = fprs.contract_date;

UPDATE full_parent_regression_selected
SET lead_arrangers_count = CARDINALITY(lead_arrangers_id_list);

UPDATE full_parent_regression_selected fprs
SET all_lenders_id_list = l.all_lenders_id_list
FROM list_a l
WHERE l.borrower_id = fprs.borrower_id
  AND l.tranche_active_date = fprs.contract_date;

UPDATE full_parent_regression_selected
SET all_lenders_count = CARDINALITY(all_lenders_id_list);


/* -------------------------------------------------------------------------- */
/* 11) Control variable pipeline (Compustat, XBRL, DealScan, contract CVs)      */
/* -------------------------------------------------------------------------- */
/*
This section is largely identical to your logic. The main documentation point is:
  - for Compustat and XBRL controls we choose the latest fiscal year end prior to contract_date.
  - for DealScan controls we roll up across all DealScan rows for the borrower/date.

Note
  Your original script also includes additional diagnostics and distributions. Keep those if needed.
*/

/* Selected contracts (the modeling sample spine) */

DROP TABLE IF EXISTS selected_contracts;

CREATE TABLE selected_contracts AS
SELECT
    cik,
    gvkey,
    contract_date,
    borrower_id
FROM full_parent_regression_selected;

CREATE INDEX IF NOT EXISTS selected_contracts_cik_contractdate_idx
    ON selected_contracts (cik, contract_date);


COMMIT;

/* -------------------------------------------------------------------------- */
/* 12) Diagnostics and descriptive summaries                                   */
/* -------------------------------------------------------------------------- */

/* Yearly summary of modeling sample (contracts per year and DV rate). */

DROP TABLE IF EXISTS yearly_summary;

CREATE TABLE yearly_summary AS
SELECT
    EXTRACT(YEAR FROM contract_date) AS contract_year,
    COUNT(*)                         AS contract_count,
    AVG(dv::INTEGER)                 AS dv_mean
FROM full_parent_regression_matches
GROUP BY contract_year
ORDER BY contract_year ASC;


/* Simple contract detail list (useful for audit trails). */

DROP TABLE IF EXISTS contract_details;

CREATE TABLE contract_details AS
SELECT
    cik,
    contract_date,
    file_link
FROM full_parent_regression_matches
ORDER BY contract_date ASC;


/* XBRL tag distribution by fiscal year (coverage diagnostic). */

DROP TABLE IF EXISTS xbrl_dist;

CREATE TABLE xbrl_dist AS
SELECT DISTINCT
    EXTRACT(YEAR FROM fiscal_year_end_date) AS fiscal_year,
    tag_title,
    COUNT(DISTINCT cik)                     AS unique_firm_count
FROM xbrl
GROUP BY fiscal_year, tag_title
ORDER BY fiscal_year, tag_title;


/* -------------------------------------------------------------------------- */
/* 13) Control variable distributions: Compustat                               */
/* -------------------------------------------------------------------------- */

CREATE INDEX IF NOT EXISTS compustat_gvkey_datadate_idx
    ON compustat (gvkey, datadate);

CREATE INDEX IF NOT EXISTS xbrl_cik_fy_filing_idx
    ON xbrl (cik, fiscal_year_end_date, form_10k_filing_date);


/*
Compustat controls
  For each (gvkey, contract_date), select the most recent Compustat annual observation prior to contract_date.

Scaling
  at_mil is total assets in $MM (at is assumed in $MM in Compustat; keep as provided in your table).
  Ratios are set to 0 when AT is missing or 0 to avoid infinities.
*/

DROP TABLE IF EXISTS compustat_cv;

CREATE TABLE compustat_cv AS
SELECT
    sc.cik,
    sc.gvkey,
    sc.contract_date,
    cs1.datadate AS fs_year_end_comp,
    COALESCE(cs1.at, 0) AS at_mil,

    CASE WHEN cs1.at IS NULL OR cs1.at = 0 THEN 0 ELSE COALESCE(cs1.ib,    0) / cs1.at END AS prof_ib_at,
    CASE WHEN cs1.at IS NULL OR cs1.at = 0 THEN 0 ELSE COALESCE(cs1.ppent, 0) / cs1.at END AS ppent_at,
    CASE WHEN cs1.at IS NULL OR cs1.at = 0 THEN 0 ELSE COALESCE(cs1.dlc,   0) / cs1.at END AS dlc_at,
    CASE WHEN cs1.at IS NULL OR cs1.at = 0 THEN 0 ELSE COALESCE(cs1.dltt,  0) / cs1.at END AS dltt_at,
    CASE
        WHEN cs1.at IS NULL OR cs1.at = 0 THEN 0
        ELSE (
            COALESCE(cs1.xrent, 0) + COALESCE(cs1.mrc1, 0) + COALESCE(cs1.mrc2, 0) + COALESCE(cs1.mrc3, 0)
          + COALESCE(cs1.mrc4, 0) + COALESCE(cs1.mrc5, 0) + COALESCE(cs1.mrcta, 0)
        ) / cs1.at
    END AS obs_lease_at
FROM selected_contracts sc
LEFT JOIN LATERAL (
    SELECT *
    FROM compustat cs
    WHERE cs.gvkey = sc.gvkey
      AND cs.datadate < sc.contract_date
    ORDER BY cs.datadate DESC
    LIMIT 1
) cs1 ON TRUE;


/* -------------------------------------------------------------------------- */
/* 14) Control variable distributions: XBRL                                   */
/* -------------------------------------------------------------------------- */

/*
XBRL controls
  - Find the latest fiscal year end before contract_date.
  - Keep that FY's XBRL facts.
  - Map tag_title to a smaller set of root tags.
  - If a root tag appears multiple times, keep the value with the latest 10-K filing date.
  - Pivot roots into columns and compute scaled measures.
*/

DROP TABLE IF EXISTS xbrl_cv;

CREATE TABLE xbrl_cv AS
WITH base AS (
    SELECT sc.cik, sc.contract_date, cc.at_mil
    FROM selected_contracts sc
    LEFT JOIN compustat_cv cc
      ON cc.gvkey = sc.gvkey
     AND cc.contract_date = sc.contract_date
),
latest_fy AS (
    SELECT
        b.cik,
        b.contract_date,
        MAX(x.fiscal_year_end_date) AS fy_end
    FROM base b
    JOIN xbrl x
      ON x.cik = b.cik
     AND x.fiscal_year_end_date < b.contract_date
    GROUP BY b.cik, b.contract_date
),
tagged AS (
    SELECT
        lf.cik,
        lf.contract_date,
        b.at_mil,
        CASE
            WHEN x.tag_title ILIKE 'FinanceLeaseLiability%'                        THEN 'FinanceLeaseLiability'
            WHEN x.tag_title ILIKE 'OperatingLeaseLiability%'                      THEN 'OperatingLeaseLiability'
            WHEN x.tag_title ILIKE 'CapitalLeaseObligations%'                      THEN 'CapitalLeaseObligations'
            WHEN x.tag_title ILIKE 'CapitalLeasesFutureMinimumPaymentsDue%'        THEN 'CapitalLeasesFutureMinimumPaymentsDue'
            WHEN x.tag_title ILIKE 'OperatingLeasesFutureMinimumPaymentsDue%'      THEN 'OperatingLeasesFutureMinimumPaymentsDue'
            WHEN x.tag_title ILIKE 'LeaseCost%'                                    THEN 'LeaseCost'
            WHEN x.tag_title ILIKE 'LeaseAndRentalExpense%'                        THEN 'LeaseAndRentalExpense'
            WHEN x.tag_title ILIKE 'VariableLeaseCost%'                            THEN 'VariableLeaseCost'
            WHEN x.tag_title ILIKE 'CapitalLeasesContingentRentalPaymentsDue%'     THEN 'CapitalLeasesContingentRentalPaymentsDue'
            WHEN x.tag_title ILIKE 'OperatingLeasesRentExpenseContingentRentals%'  THEN 'OperatingLeasesRentExpenseContingentRentals'
            ELSE NULL
        END AS tag_root,
        (x.dollar_value::NUMERIC / 1e6) AS value_mil,
        x.form_10k_filing_date
    FROM latest_fy lf
    JOIN xbrl x
      ON x.cik = lf.cik
     AND x.fiscal_year_end_date = lf.fy_end
    JOIN base b
      ON b.cik = lf.cik
     AND b.contract_date = lf.contract_date
),
dedup AS (
    SELECT cik, contract_date, at_mil, tag_root, value_mil
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY cik, contract_date, tag_root
                ORDER BY form_10k_filing_date DESC
            ) AS rn
        FROM tagged
        WHERE tag_root IS NOT NULL
    ) q
    WHERE rn = 1
),
pivot AS (
    SELECT
        cik,
        contract_date,
        MAX(at_mil) AS at_mil,
        MAX(value_mil) FILTER (WHERE tag_root = 'FinanceLeaseLiability')                        AS finance_lease_liab,
        MAX(value_mil) FILTER (WHERE tag_root = 'OperatingLeaseLiability')                      AS operating_lease_liab,
        MAX(value_mil) FILTER (WHERE tag_root = 'CapitalLeaseObligations')                      AS capital_lease_oblig,
        MAX(value_mil) FILTER (WHERE tag_root = 'CapitalLeasesFutureMinimumPaymentsDue')        AS capital_lease_min_pay,
        MAX(value_mil) FILTER (WHERE tag_root = 'OperatingLeasesFutureMinimumPaymentsDue')      AS operating_min_pay,
        MAX(value_mil) FILTER (WHERE tag_root = 'LeaseCost')                                    AS lease_cost,
        MAX(value_mil) FILTER (WHERE tag_root = 'LeaseAndRentalExpense')                        AS lease_rent_expense,
        MAX(value_mil) FILTER (WHERE tag_root = 'VariableLeaseCost')                            AS variable_lease_cost,
        MAX(value_mil) FILTER (WHERE tag_root = 'CapitalLeasesContingentRentalPaymentsDue')     AS capital_contingent,
        MAX(value_mil) FILTER (WHERE tag_root = 'OperatingLeasesRentExpenseContingentRentals')  AS operating_contingent
    FROM dedup
    GROUP BY cik, contract_date
)
SELECT
    p.*,
    CASE
        WHEN COALESCE(p.at_mil, 0) = 0 THEN 0
        WHEN COALESCE(p.finance_lease_liab, 0) + COALESCE(p.operating_lease_liab, 0) <> 0
            THEN (COALESCE(p.finance_lease_liab, 0) + COALESCE(p.operating_lease_liab, 0)) / p.at_mil
        WHEN COALESCE(p.capital_lease_oblig, 0) <> 0
            THEN p.capital_lease_oblig / p.at_mil
        ELSE COALESCE(p.capital_lease_min_pay, 0) / p.at_mil
    END AS xbrl_onbs,
    CASE WHEN COALESCE(p.at_mil, 0) = 0 THEN 0 ELSE COALESCE(p.operating_min_pay, 0) / p.at_mil END AS xbrl_offbs,
    CASE
        WHEN COALESCE(p.at_mil, 0) = 0 THEN 0
        WHEN COALESCE(p.lease_cost, 0) <> 0 THEN p.lease_cost / p.at_mil
        ELSE COALESCE(p.lease_rent_expense, 0) / p.at_mil
    END AS xbrl_is,
    CASE
        WHEN COALESCE(p.at_mil, 0) = 0 THEN 0
        ELSE (
            COALESCE(p.variable_lease_cost, 0)
          + COALESCE(p.capital_contingent, 0)
          + COALESCE(p.operating_contingent, 0)
        ) / p.at_mil
    END AS xbrl_vl
FROM pivot p;


/* -------------------------------------------------------------------------- */
/* 15) Contract-derived controls (from DV covenant extraction output)          */
/* -------------------------------------------------------------------------- */

/*
Input
  dv_contracts_cv_selected is assumed to be provided upstream as a post-processed table.
  This script does not rebuild the DV contract tables or run DV cross-checks.

Output
  contracts_cv: contract-level controls scaled by Compustat assets.
*/

DROP TABLE IF EXISTS contracts_cv;

CREATE TABLE contracts_cv AS
SELECT
    sc.*,
    cc.at_mil,
    dcc.frozen_gaap,
    dcc.renegotiation_for_gaap_change,
    dcc.not_adopt_asc,
    CASE WHEN cc.at_mil IS NULL OR cc.at_mil = 0 THEN 0 ELSE COALESCE(dcc.limit_$, 0) / (cc.at_mil * 1e6) END AS debt_covenant,
    CASE WHEN cc.at_mil IS NULL OR cc.at_mil = 0 THEN 0 ELSE COALESCE(dcc.max_covenant_value, 0) / (cc.at_mil * 1e6) END AS lease_covenant
FROM selected_contracts sc
LEFT JOIN compustat_cv cc
  ON cc.cik = sc.cik
 AND cc.contract_date = sc.contract_date
LEFT JOIN dv_contracts_cv_selected dcc
  ON dcc.cik = sc.cik
 AND dcc.contract_date = sc.contract_date;


/* -------------------------------------------------------------------------- */
/* 16) DealScan controls (roll up across DealScan rows per borrower/date)      */
/* -------------------------------------------------------------------------- */

/* Clean numeric covenant fields that arrive as text. */

ALTER TABLE dealscan
ALTER COLUMN tangible_net_worth TYPE BIGINT
USING (
    CASE
        WHEN NULLIF(TRIM(tangible_net_worth), '') IS NULL THEN NULL
        ELSE ROUND(REGEXP_REPLACE(tangible_net_worth, '[^0-9eE+.\-]', '', 'g')::NUMERIC)::BIGINT
    END
);

ALTER TABLE dealscan
ALTER COLUMN net_worth TYPE BIGINT
USING (
    CASE
        WHEN NULLIF(TRIM(net_worth), '') IS NULL THEN NULL
        ELSE ROUND(REGEXP_REPLACE(net_worth, '[^0-9eE+.\-]', '', 'g')::NUMERIC)::BIGINT
    END
);


/* Restrict LISTA/LISTB to the selected contract set to speed downstream joins. */

DROP TABLE IF EXISTS lista_selected;

CREATE TABLE lista_selected AS
SELECT *
FROM list_a
WHERE (tranche_active_date, borrower_id) IN (
    SELECT contract_date, borrower_id
    FROM selected_contracts
);


/* Past relationship frequency: lender-borrower relationships in the prior 36 months. */

DROP TABLE IF EXISTS past_relationships;

CREATE TABLE past_relationships AS
SELECT
    a.borrower_id,
    a.tranche_active_date,
    COUNT(*) FILTER (WHERE a.borrower_id = b.borrower_id) AS relationship_count
FROM lista_selected a
LEFT JOIN list_b b
  ON b.lender_parent_id = a.lender_parent_id
 AND b.tranche_active_date BETWEEN (a.tranche_active_date - INTERVAL '36 months')
                              AND (a.tranche_active_date - INTERVAL '1 day')
GROUP BY a.borrower_id, a.tranche_active_date
ORDER BY a.borrower_id, a.tranche_active_date;


CREATE INDEX IF NOT EXISTS contracts_selected_borrower_date_idx
    ON lista_selected (borrower_id, tranche_active_date, lpc_deal_id);

CREATE INDEX IF NOT EXISTS contract_lenders_contract_lender_idx
    ON list_b (lpc_deal_id, lender_parent_id);


/*
DealScan CV rollup
  - Build covenant presence flags per DealScan row.
  - Roll up to (borrower_id, tranche_active_date) using:
      * tranche_amount: SUM
      * maturity: MAX
      * interest_rate: tranche-amount weighted average
      * covenant indicators: MAX
      * covenant_count: sum of MAX flags (presence of each covenant type)
*/

DROP TABLE IF EXISTS dealscan_cv;

CREATE TABLE dealscan_cv AS
WITH base AS (
    SELECT
        sc.borrower_id,
        sc.contract_date AS tranche_active_date,
        d.tranche_amount_converted,
        d.tranche_maturity_date,
        d.all_in_spread_drawn_bps,
        d.seniority_type,
        d.secured,
        d.market_segment,
        d.performance_pricing_grid,
        d.max_leverage_ratio_ds,
        d.max_debt_to_cash_flow,
        d.max_sr_debt_to_cash_flow,
        d.max_debt_to_tangible_net_worth,
        d.max_debt_to_equity_ratio,
        d.max_loan_to_value_ratio,
        d.min_fixed_charge_coverage_ratio,
        d.min_debt_service_coverage_ratio,
        d.min_interest_coverage_ratio,
        d.min_cash_interest_coverage_ratio,
        d.min_current_ratio,
        d.tangible_net_worth,
        d.net_worth,
        d.excess_cf_sweep,
        d.asset_sales_sweep,
        d.debt_issue_sweep,
        d.equity_issue_sweep,
        d.insurance_proceeds_sweep,
        d.percentage_of_net_income,
        d.collateral_release,
        d.required_lenders,
        d.terms_changes,
        d.ebitda_initial_amount_usd,
        d.ebitda_final_amount_usd,
        d.capex_initial_usd,
        d.capex_final_usd
    FROM selected_contracts sc
    LEFT JOIN dealscan d
      ON d.borrower_id = sc.borrower_id
     AND d.tranche_active_date = sc.contract_date
),
maturity_row AS (
    SELECT
        b.*,
        CASE
            WHEN b.tranche_maturity_date IS NULL THEN NULL
            ELSE (
                EXTRACT(YEAR FROM AGE(b.tranche_maturity_date, b.tranche_active_date)) * 12
              + EXTRACT(MONTH FROM AGE(b.tranche_maturity_date, b.tranche_active_date))
            )::INT
        END AS maturity_months_raw
    FROM base b
),
cov_flags AS (
    SELECT
        m.*,
        CASE WHEN m.max_leverage_ratio_ds <> '0' THEN 1 ELSE 0 END AS f_max_leverage_ratio,
        CASE WHEN m.max_debt_to_cash_flow <> '0' THEN 1 ELSE 0 END AS f_max_debt_to_cash_flow,
        CASE WHEN m.max_sr_debt_to_cash_flow <> '0' THEN 1 ELSE 0 END AS f_max_sr_debt_to_cash_flow,
        CASE WHEN m.max_debt_to_tangible_net_worth <> '0' THEN 1 ELSE 0 END AS f_max_debt_to_tnw,
        CASE WHEN m.max_debt_to_equity_ratio <> '0' THEN 1 ELSE 0 END AS f_max_debt_to_equity,
        CASE WHEN m.max_loan_to_value_ratio <> '0' THEN 1 ELSE 0 END AS f_max_ltv,
        CASE WHEN m.min_fixed_charge_coverage_ratio <> '0' THEN 1 ELSE 0 END AS f_min_fccr,
        CASE WHEN m.min_debt_service_coverage_ratio <> '0' THEN 1 ELSE 0 END AS f_min_dscr,
        CASE WHEN m.min_interest_coverage_ratio <> '0' THEN 1 ELSE 0 END AS f_min_icr,
        CASE WHEN m.min_cash_interest_coverage_ratio <> '0' THEN 1 ELSE 0 END AS f_min_cicr,
        CASE WHEN m.min_current_ratio <> '0' THEN 1 ELSE 0 END AS f_min_current,
        CASE WHEN COALESCE(m.tangible_net_worth::NUMERIC, 0) <> 0 THEN 1 ELSE 0 END AS f_tnw,
        CASE WHEN COALESCE(m.net_worth::NUMERIC, 0) <> 0 THEN 1 ELSE 0 END AS f_nw,
        CASE WHEN COALESCE(m.excess_cf_sweep::NUMERIC, 0) <> 0 THEN 1 ELSE 0 END AS f_excess_cf_sweep,
        CASE WHEN COALESCE(m.asset_sales_sweep::NUMERIC, 0) <> 0 THEN 1 ELSE 0 END AS f_asset_sales_sweep,
        CASE WHEN COALESCE(m.debt_issue_sweep::NUMERIC, 0) <> 0 THEN 1 ELSE 0 END AS f_debt_issue_sweep,
        CASE WHEN COALESCE(m.equity_issue_sweep::NUMERIC, 0) <> 0 THEN 1 ELSE 0 END AS f_equity_issue_sweep,
        CASE WHEN COALESCE(m.insurance_proceeds_sweep::NUMERIC, 0) <> 0 THEN 1 ELSE 0 END AS f_insurance_proceeds_sweep,
        CASE WHEN COALESCE(m.percentage_of_net_income::NUMERIC, 0) <> 0 THEN 1 ELSE 0 END AS f_pct_net_income,
        CASE WHEN COALESCE(m.collateral_release::NUMERIC, 0) <> 0 THEN 1 ELSE 0 END AS f_collateral_release,
        CASE WHEN COALESCE(m.required_lenders::NUMERIC, 0) <> 0 THEN 1 ELSE 0 END AS f_required_lenders,
        CASE WHEN COALESCE(m.terms_changes::NUMERIC, 0) <> 0 THEN 1 ELSE 0 END AS f_terms_changes,
        CASE WHEN COALESCE(m.ebitda_initial_amount_usd::NUMERIC, 0) <> 0 THEN 1 ELSE 0 END AS f_ebitda_init,
        CASE WHEN COALESCE(m.ebitda_final_amount_usd::NUMERIC, 0) <> 0 THEN 1 ELSE 0 END AS f_ebitda_final,
        CASE WHEN COALESCE(m.capex_initial_usd::NUMERIC, 0) <> 0 THEN 1 ELSE 0 END AS f_capex_init,
        CASE WHEN COALESCE(m.capex_final_usd::NUMERIC, 0) <> 0 THEN 1 ELSE 0 END AS f_capex_final
    FROM maturity_row m
),
contract_rollup AS (
    SELECT
        c.borrower_id,
        c.tranche_active_date,
        SUM(COALESCE(c.tranche_amount_converted, 0)) AS tranche_amount_converted,
        MAX(c.maturity_months_raw)                   AS maturity,
        CASE
            WHEN SUM(COALESCE(c.tranche_amount_converted, 0)) = 0 THEN NULL
            ELSE SUM(COALESCE(c.all_in_spread_drawn_bps, 0) * COALESCE(c.tranche_amount_converted, 0))
               / SUM(COALESCE(c.tranche_amount_converted, 0))
        END AS interest_rate,
        MAX((LOWER(TRIM(c.seniority_type)) <> 'senior')::INT)          AS second_or_lower_lien,
        MAX((LOWER(TRIM(c.secured)) = 'no')::INT)                      AS unsecured,
        MAX((c.market_segment ILIKE '%non investment grade%')::INT)    AS non_investment_grade,
        MAX((c.performance_pricing_grid = 'yes')::INT)                 AS performance_pricing,
        GREATEST(
            MAX(f_max_leverage_ratio),
            MAX(f_max_debt_to_cash_flow),
            MAX(f_max_sr_debt_to_cash_flow),
            MAX(f_max_debt_to_tnw),
            MAX(f_max_debt_to_equity),
            MAX(f_max_ltv)
        ) AS max_debt_covenant,
        GREATEST(
            MAX(f_min_fccr),
            MAX(f_min_dscr),
            MAX(f_min_icr),
            MAX(f_min_cicr),
            MAX(f_min_current)
        ) AS min_liquid_covenant,
        (
            MAX(f_max_leverage_ratio) +
            MAX(f_max_debt_to_cash_flow) +
            MAX(f_max_sr_debt_to_cash_flow) +
            MAX(f_max_debt_to_tnw) +
            MAX(f_max_debt_to_equity) +
            MAX(f_max_ltv) +
            MAX(f_min_fccr) +
            MAX(f_min_dscr) +
            MAX(f_min_icr) +
            MAX(f_min_cicr) +
            MAX(f_min_current) +
            MAX(f_tnw) +
            MAX(f_nw) +
            MAX(f_excess_cf_sweep) +
            MAX(f_asset_sales_sweep) +
            MAX(f_debt_issue_sweep) +
            MAX(f_equity_issue_sweep) +
            MAX(f_insurance_proceeds_sweep) +
            MAX(f_pct_net_income) +
            MAX(f_collateral_release) +
            MAX(f_required_lenders) +
            MAX(f_terms_changes) +
            MAX(f_ebitda_init) +
            MAX(f_ebitda_final) +
            MAX(f_capex_init) +
            MAX(f_capex_final)
        ) AS covenant_count
    FROM cov_flags c
    GROUP BY c.borrower_id, c.tranche_active_date
)
SELECT
    cr.borrower_id,
    cr.tranche_active_date,
    cr.tranche_amount_converted,
    cr.maturity,
    cr.interest_rate,
    fprs.all_lenders_count,
    LEAST(
        1::NUMERIC,
        COALESCE(pr.relationship_count::NUMERIC, 0)
        / NULLIF(fprs.all_lenders_count::NUMERIC, 0)
    ) AS relationship_freq,
    cr.second_or_lower_lien::INT AS second_or_lower_lien,
    cr.unsecured::INT            AS unsecured,
    cr.non_investment_grade::INT AS non_investment_grade,
    cr.performance_pricing::INT  AS performance_pricing,
    cr.covenant_count::INT       AS covenant_count,
    cr.max_debt_covenant::INT    AS max_debt_covenant,
    cr.min_liquid_covenant::INT  AS min_liquid_covenant
FROM contract_rollup cr
LEFT JOIN selected_contracts sc
  ON sc.borrower_id = cr.borrower_id
 AND sc.contract_date = cr.tranche_active_date
LEFT JOIN full_parent_regression_selected fprs
  ON fprs.cik = sc.cik
 AND fprs.contract_date = cr.tranche_active_date
LEFT JOIN past_relationships pr
  ON pr.borrower_id = cr.borrower_id
 AND pr.tranche_active_date = cr.tranche_active_date
ORDER BY cr.borrower_id, cr.tranche_active_date;


/* Keep your original maturity imputation rule (global max). */

UPDATE dealscan_cv dc
SET maturity = t.max_maturity
FROM (SELECT MAX(maturity) AS max_maturity FROM dealscan_cv) t
WHERE dc.maturity IS NULL;


/* -------------------------------------------------------------------------- */
/* 17) Assemble contract controls                                              */
/* -------------------------------------------------------------------------- */

DROP TABLE IF EXISTS contract_controls;

CREATE TABLE contract_controls AS
SELECT
    sc.*,
    cc.fs_year_end_comp,
    COALESCE(cc.at_mil, 0)               AS at_mil,
    COALESCE(cc.prof_ib_at, 0)           AS prof_ib_at,
    COALESCE(cc.ppent_at, 0)             AS ppent_at,
    COALESCE(cc.dlc_at + cc.dltt_at, 0)  AS dlc_dltt_at,
    COALESCE(cc.obs_lease_at, 0)         AS obs_lease_at,
    COALESCE(xb.xbrl_onbs, 0)            AS xbrl_onbs,
    COALESCE(xb.xbrl_offbs, 0)           AS xbrl_offbs,
    COALESCE(xb.xbrl_is, 0)              AS xbrl_is,
    COALESCE(xb.xbrl_vl, 0)              AS xbrl_vl,
    COALESCE(cc2.frozen_gaap, B'0')      AS frozen_gaap,
    COALESCE(cc2.renegotiation_for_gaap_change, B'0') AS renegotiation_for_gaap_change,
    COALESCE(cc2.not_adopt_asc, B'0')    AS not_adopt_asc,
    COALESCE(cc2.debt_covenant, 0)       AS debt_covenant,
    COALESCE(cc2.lease_covenant, 0)      AS lease_covenant,
    COALESCE(dc.tranche_amount_converted, 0) AS tranche_amount_converted,
    COALESCE(dc.maturity, 0)             AS maturity,
    COALESCE(dc.interest_rate, 0)        AS interest_rate,
    COALESCE(dc.all_lenders_count, 0)    AS all_lenders_count,
    COALESCE(dc.relationship_freq, 0)    AS relationship_freq,
    COALESCE(dc.second_or_lower_lien, 0) AS second_or_lower_lien,
    COALESCE(dc.unsecured, 0)            AS unsecured,
    COALESCE(dc.non_investment_grade, 0) AS non_investment_grade,
    COALESCE(dc.performance_pricing, 0)  AS performance_pricing,
    COALESCE(dc.covenant_count, 0)       AS covenant_count,
    COALESCE(dc.max_debt_covenant, 0)    AS max_debt_covenant,
    COALESCE(dc.min_liquid_covenant, 0)  AS min_liquid_covenant
FROM selected_contracts sc
LEFT JOIN compustat_cv cc
  ON cc.cik = sc.cik
 AND cc.gvkey = sc.gvkey
 AND cc.contract_date = sc.contract_date
LEFT JOIN xbrl_cv xb
  ON xb.cik = sc.cik
 AND xb.contract_date = sc.contract_date
LEFT JOIN contracts_cv cc2
  ON cc2.cik = sc.cik
 AND cc2.contract_date = sc.contract_date
LEFT JOIN dealscan_cv dc
  ON dc.borrower_id = sc.borrower_id
 AND dc.tranche_active_date = sc.contract_date;


/* -------------------------------------------------------------------------- */
/* 18) Final modeling table (DV + IV + controls)                               */
/* -------------------------------------------------------------------------- */

DROP TABLE IF EXISTS full_model_data;

CREATE TABLE full_model_data AS
SELECT
    fprs.*, 
    cc.at_mil,
    cc.prof_ib_at,
    cc.ppent_at,
    cc.dlc_dltt_at,
    cc.obs_lease_at,
    cc.xbrl_onbs,
    cc.xbrl_offbs,
    cc.xbrl_is,
    cc.xbrl_vl,
    cc.frozen_gaap,
    cc.renegotiation_for_gaap_change,
    cc.not_adopt_asc,
    cc.debt_covenant,
    cc.lease_covenant,
    cc.tranche_amount_converted,
    cc.maturity,
    cc.interest_rate,
    cc.relationship_freq,
    cc.second_or_lower_lien,
    cc.unsecured,
    cc.non_investment_grade,
    cc.performance_pricing,
    cc.covenant_count,
    cc.max_debt_covenant,
    cc.min_liquid_covenant
FROM full_parent_regression_selected fprs
LEFT JOIN contract_controls cc
  ON fprs.cik = cc.cik
 AND fprs.contract_date = cc.contract_date;


COMMIT;

/* End of script (regression done outside SQL). */
