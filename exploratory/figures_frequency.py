"""
Figures for Ayung:
  1. Contract frequency by year — raw counts, broken out by each GenAI score category
  2. Contract frequency by year — % share within year, broken out by each GenAI score category
  3. Cross-correlation among GenAI scores by year (heatmap grid)

Data source: Ayung's RA LLM-scored dataset (Box CSV)
"""

import math
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO    = Path(__file__).resolve().parent.parent  # code/ — this script lives in code/exploratory/
CACHE   = REPO.parent / "data" / "contracts_dv.parquet"
BOX_URL = "https://ucdavis.box.com/shared/static/viblqy8rovvqlredda5s84kcrebctyyx.csv"
OUT_DIR = REPO.parent / "output" / "figures" / "ayung_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load / cache data ──────────────────────────────────────────────────────────
if CACHE.exists():
    print(f"Loading cached data from {CACHE}")
    df = pd.read_parquet(CACHE)
else:
    print("Downloading from Box …")
    resp = requests.get(BOX_URL, timeout=120)
    resp.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    print(f"Cached {len(df):,} rows to {CACHE}")

# ── Sample filters ─────────────────────────────────────────────────────────────
df = df[df["claude_is_debt_contract"] == "Y"].copy()
df = df[df["claude_contract_type"] != "Non-debt"].copy()
df["year"] = pd.to_datetime(df["tranche_active_date"]).dt.year
df = df[(df["year"] >= 2000) & (df["year"] <= 2024)].copy()

print(f"Analysis sample: {len(df):,} obs, {df['year'].min()}–{df['year'].max()}")

# ── Score variables ────────────────────────────────────────────────────────────
SCORES = {
    "SLB":           "claude_SLB_SCORE",           # Sale-leaseback (0–3)
    "SYN":           "claude_SYN_SCORE",           # Synthetic lease (0–3)
    "OPL":           "claude_OPL_SCORE",           # Operating lease (0–3)
    "VAR":           "claude_VAR_SCORE",           # Variable lease (0–2)
    "RES":           "claude_RES_SCORE",           # Residual value (0–2)
    "GAAP Override": "claude_GAAP_OVERRIDE_SCORE", # GAAP override (0–2)
    "Freeze":        "claude_FREEZE_SCORE",        # Freeze clause (0–2)
}

for col in SCORES.values():
    df[col] = pd.to_numeric(df[col], errors="coerce")

years = sorted(df["year"].unique())

SCORE_COLORS = {
    0: "#d9e8f5",
    1: "#4a90d9",
    2: "#1a5fa8",
    3: "#0c2e5e",
}

# ── Shared helper: build stacked bar subplots ──────────────────────────────────
def make_bar_figure(pct: bool) -> plt.Figure:
    """Draw one 7-panel stacked bar figure. pct=True normalises bars to 100%."""
    n     = len(SCORES)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows),
                             constrained_layout=True)
    axes_flat = axes.flatten()

    for ax_idx, (label, col) in enumerate(SCORES.items()):
        ax  = axes_flat[ax_idx]
        sub = df[["year", col]].dropna()

        pivot = (sub.groupby(["year", col])
                   .size()
                   .unstack(fill_value=0)
                   .reindex(years, fill_value=0))

        score_levels = sorted(pivot.columns.astype(int))
        pivot = pivot[[c for c in score_levels if c in pivot.columns]]

        if pct:
            pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100

        colors = [SCORE_COLORS.get(int(c), "#888888") for c in pivot.columns]
        pivot.plot(kind="bar", stacked=True, ax=ax,
                   color=colors, edgecolor="none", width=0.8, legend=True)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.legend(title="Score", fontsize=7, title_fontsize=7,
                  loc="upper left", framealpha=0.7)

        if pct:
            ax.set_ylabel("% of Contracts")
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
            label_y = 92
        else:
            ax.set_ylabel("# Contracts")
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            label_y = ax.get_ylim()[1] * 0.92

        ax2019 = years.index(2019) if 2019 in years else None
        if ax2019 is not None:
            ax.axvline(ax2019 - 0.5, color="red", lw=1, ls="--", alpha=0.7)
            ax.text(ax2019 - 0.4, label_y, "ASC 842",
                    color="red", fontsize=6, va="top")

    for i in range(len(SCORES), len(axes_flat)):
        axes_flat[i].set_visible(False)

    return fig

# ── Figure 1: Raw counts ───────────────────────────────────────────────────────
fig1 = make_bar_figure(pct=False)
fig1.suptitle("Contract Frequency by Year and GenAI Score Category\n"
              "(Debt contracts only; dashed line = ASC 842 effective 2019)",
              fontsize=12, fontweight="bold")
out1 = OUT_DIR / "fig1_frequency_counts_by_year.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved Figure 1 → {out1}")

# ── Figure 2: Percentage share ─────────────────────────────────────────────────
fig2 = make_bar_figure(pct=True)
fig2.suptitle("Contract Frequency by Year and GenAI Score Category (% Share)\n"
              "(Debt contracts only; dashed line = ASC 842 effective 2019)",
              fontsize=12, fontweight="bold")
out2 = OUT_DIR / "fig2_frequency_pct_by_year.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved Figure 2 → {out2}")

# ── Figure 3: Cross-correlation heatmaps by year ──────────────────────────────
score_cols  = list(SCORES.values())
score_names = list(SCORES.keys())

n_years = len(years)
ncols3  = 5
nrows3  = math.ceil(n_years / ncols3)

fig3, axes3 = plt.subplots(nrows3, ncols3,
                            figsize=(ncols3 * 2.8, nrows3 * 2.8),
                            constrained_layout=True)
axes3_flat = axes3.flatten()

def _stars(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""

n = len(score_names)

for i, yr in enumerate(years):
    ax  = axes3_flat[i]
    sub = df[df["year"] == yr][score_cols].dropna()

    # Spearman correlation matrix + p-value matrix
    res  = stats.spearmanr(sub)
    rmat = np.array(res.statistic) if sub.shape[1] > 2 else np.array([[1, res.statistic], [res.statistic, 1]])
    pmat = np.array(res.pvalue)    if sub.shape[1] > 2 else np.array([[0, res.pvalue],    [res.pvalue,    0]])

    ax.imshow(np.ones((n, n)), vmin=0, vmax=1, cmap="Greys", aspect="auto", alpha=0)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    # Draw grid lines manually
    for k in range(n + 1):
        ax.axhline(k - 0.5, color="#dddddd", linewidth=0.4)
        ax.axvline(k - 0.5, color="#dddddd", linewidth=0.4)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(score_names, rotation=90, fontsize=5.5)
    ax.set_yticklabels(score_names, fontsize=5.5)
    ax.set_title(str(yr), fontsize=8,
                 fontweight="bold" if yr == 2019 else "normal",
                 color="red" if yr == 2019 else "black")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    for r in range(n):
        for c in range(n):
            val = rmat[r, c]
            p   = pmat[r, c]
            if r == c:
                # diagonal: shade lightly
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           color="#f0f0f0", zorder=0))
                ax.text(c, r, "1.00", ha="center", va="center",
                        fontsize=4, color="#888888")
            elif not math.isnan(val):
                stars = _stars(p)
                ax.text(c, r, f"{val:.2f}{stars}", ha="center", va="center",
                        fontsize=4, color="black")

for i in range(n_years, len(axes3_flat)):
    axes3_flat[i].set_visible(False)

fig3.suptitle("Cross-Correlation Among GenAI Scores by Year (Spearman ρ)\n"
              "(*** p<0.01  ** p<0.05  * p<0.10 | 2019 = ASC 842 effective, shown in red)",
              fontsize=11, fontweight="bold")
out3 = OUT_DIR / "fig3_crosscorr_by_year.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved Figure 3 → {out3}")

plt.close("all")
print("Done.")
