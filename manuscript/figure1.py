"""
figure1.py  —  Figure 1: contract frequency by year and GenAI score category (% share)
=======================================================================================

Seven stacked-bar panels, one per GenAI score, showing the WITHIN-YEAR PERCENTAGE SHARE of
each score level across 2000–2024. A dashed vertical marker at 2019 flags ASC 842 taking
effect. Bars are normalised to 100% within each year, so the panels show composition rather
than volume — year-to-year changes in contract counts do not distort the picture.

Isolated from exploratory/figures_frequency.py, which produced three figures; this keeps
only its Figure 2 (`fig2_frequency_pct_by_year.png`), which is the manuscript's Figure 1.
The raw-count and cross-correlation figures, and the `math`/`scipy` imports they needed, are
dropped. The shared bar-drawing helper is specialised to the percentage form.

DATA. Reads the LLM-scored contract dataset (contracts_dv.parquet), NOT the merged
analysis file — this figure describes the scored corpus, so it is not restricted to the
regression estimation sample and its N is much larger than the tables'. Falls back to
downloading from Box and caching if the parquet is absent.

Input : data/contracts_dv.parquet  (or the Box CSV on first run)
Output: output/figures/figure1.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_DIR = Path(__file__).resolve().parent.parent   # code/ — this script lives in code/manuscript/
DATA_DIR = REPO_DIR.parent / "data"
FIG_DIR  = REPO_DIR.parent / "output" / "figures"
CACHE    = DATA_DIR / "contracts_dv.parquet"
OUT_FILE = FIG_DIR / "figure1.png"
BOX_URL  = "https://ucdavis.box.com/shared/static/viblqy8rovvqlredda5s84kcrebctyyx.csv"

YEAR_MIN, YEAR_MAX = 2000, 2024
ASC842_YEAR = 2019

SCORES = {
    "SLB":           "claude_SLB_SCORE",           # Sale-leaseback (0–3)
    "SYN":           "claude_SYN_SCORE",           # Synthetic lease (0–3)
    "OPL":           "claude_OPL_SCORE",           # Operating lease (0–3)
    "VAR":           "claude_VAR_SCORE",           # Variable lease (0–2)
    "RES":           "claude_RES_SCORE",           # Residual value (0–2)
    "GAAP Override": "claude_GAAP_OVERRIDE_SCORE", # GAAP override (0–2)
    "Freeze":        "claude_FREEZE_SCORE",        # Freeze clause (0–2)
}
SCORE_COLORS = {0: "#d9e8f5", 1: "#4a90d9", 2: "#1a5fa8", 3: "#0c2e5e"}


def load_scored_contracts() -> pd.DataFrame:
    """The LLM-scored corpus, cached locally; downloads from Box only if absent."""
    if CACHE.exists():
        print(f"Loading cached data from {CACHE}")
        df = pd.read_parquet(CACHE)
    else:
        print("Downloading from Box …")
        import requests
        from io import StringIO
        resp = requests.get(BOX_URL, timeout=120)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE, index=False)
        print(f"Cached {len(df):,} rows to {CACHE}")

    df = df[df["claude_is_debt_contract"] == "Y"]
    df = df[df["claude_contract_type"] != "Non-debt"].copy()
    df["year"] = pd.to_datetime(df["tranche_active_date"]).dt.year
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    for col in SCORES.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  analysis sample: {len(df):,} obs, {df['year'].min()}–{df['year'].max()}")
    return df


def make_figure(df: pd.DataFrame) -> plt.Figure:
    """Seven stacked-bar panels of within-year score composition (% of contracts)."""
    years = sorted(df["year"].unique())
    nrows = (len(SCORES) + 1) // 2

    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4 * nrows), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax_idx, (label, col) in enumerate(SCORES.items()):
        ax  = axes_flat[ax_idx]
        sub = df[["year", col]].dropna()

        pivot = (sub.groupby(["year", col]).size()
                    .unstack(fill_value=0)
                    .reindex(years, fill_value=0))
        pivot = pivot[[c for c in sorted(pivot.columns.astype(int)) if c in pivot.columns]]
        pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100          # within-year share

        pivot.plot(kind="bar", stacked=True, ax=ax, width=0.8, legend=True,
                   color=[SCORE_COLORS.get(int(c), "#888888") for c in pivot.columns],
                   edgecolor="none")

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("% of Contracts")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.legend(title="Score", fontsize=7, title_fontsize=7,
                  loc="upper left", framealpha=0.7)

        if ASC842_YEAR in years:
            xpos = years.index(ASC842_YEAR)
            ax.axvline(xpos - 0.5, color="red", lw=1, ls="--", alpha=0.7)
            ax.text(xpos - 0.4, 92, "ASC 842", color="red", fontsize=6, va="top")

    for i in range(len(SCORES), len(axes_flat)):      # hide the unused 8th cell
        axes_flat[i].set_visible(False)

    fig.suptitle("Contract Frequency by Year and GenAI Score Category (% Share)\n"
                 f"(Debt contracts only; dashed line = ASC 842 effective {ASC842_YEAR})",
                 fontsize=12, fontweight="bold")
    return fig


def run() -> None:
    df  = load_scored_contracts()
    fig = make_figure(df)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    run()
