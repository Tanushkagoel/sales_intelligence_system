"""
Part 2 – Data Exploration & Insights
SkyGeni Sales Intelligence: EDA, business insights, and custom metrics.
Run: python eda_insights.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Optional: plotting (skip if no display)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

DATA_PATH = Path(__file__).parent / "skygeni_sales_data.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["created_date"] = pd.to_datetime(df["created_date"])
    df["closed_date"] = pd.to_datetime(df["closed_date"])
    df["closed_quarter"] = df["closed_date"].dt.to_period("Q")
    df["won"] = (df["outcome"] == "Won").astype(int)
    return df


def custom_metric_pipeline_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Custom Metric 1: Pipeline Quality Score (PQS)
    For each segment (e.g. region × lead_source), we compute historical win rate.
    PQS for a segment = that win rate (0–1). Overall PQS = weighted average by deal count.
    Interpretation: How much of current pipeline is in 'traditionally winning' segments?
    """
    segment_cols = ["region", "industry", "product_type", "lead_source"]
    segment_win_rates = (
        df.groupby(segment_cols)
        .agg(wins=("won", "sum"), total=("won", "count"))
        .assign(win_rate=lambda x: x["wins"] / x["total"])
        .reset_index()
    )
    # PQS for a segment = its historical win rate
    return segment_win_rates.rename(columns={"win_rate": "pqs"})


def custom_metric_cycle_win_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Custom Metric 2: Cycle Win Efficiency (CWE)
    By segment: win_rate / (median_sales_cycle_days / global_median_days).
    High CWE = we win at a reasonable pace in this segment; low = we either lose a lot
    or take too long to win. Captures 'efficiency' of the sales motion.
    """
    global_median_cycle = df["sales_cycle_days"].median()
    segment_cols = ["region", "lead_source"]
    agg = (
        df.groupby(segment_cols)
        .agg(
            win_rate=("won", "mean"),
            median_cycle=("sales_cycle_days", "median"),
            count=("deal_id", "count"),
        )
        .reset_index()
    )
    agg["cycle_ratio"] = agg["median_cycle"] / global_median_cycle
    agg["cwe"] = agg["win_rate"] / np.clip(agg["cycle_ratio"], 0.3, 3)
    return agg


def run_eda(df: pd.DataFrame):
    """Basic EDA: shape, dtypes, missing, key stats."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df['created_date'].min()} to {df['closed_date'].max()}")
    print("\nDtypes & non-null:")
    print(df.dtypes.to_string())
    print("\nMissing values:")
    print(df.isnull().sum().to_string())
    print("\nOutcome distribution:")
    print(df["outcome"].value_counts().to_string())
    print("\nWin rate (overall):", df["won"].mean())
    print("\nKey columns - value counts:")
    for col in ["region", "industry", "product_type", "lead_source", "deal_stage"]:
        print(f"\n{col}:")
        print(df[col].value_counts().head(10).to_string())
    print("\nDeal amount (ACV) - describe:")
    print(df["deal_amount"].describe().to_string())
    print("\nSales cycle days - describe:")
    print(df["sales_cycle_days"].describe().to_string())
    return df


def insight_1_win_rate_by_region_and_quarter(df: pd.DataFrame):
    """
    Insight 1: Win rate by region and quarter — where did it drop?
    """
    wr = (
        df.groupby(["closed_quarter", "region"])
        .agg(wins=("won", "sum"), total=("won", "count"))
        .assign(win_rate=lambda x: x["wins"] / x["total"])
        .reset_index()
    )
    pivot = wr.pivot(index="closed_quarter", columns="region", values="win_rate")
    print("\n" + "=" * 60)
    print("INSIGHT 1: Win rate by region and quarter")
    print("=" * 60)
    print(pivot.round(3).to_string())
    # Quarter-over-quarter drop
    if len(pivot) >= 2:
        latest = pivot.iloc[-1]
        prior = pivot.iloc[-2]
        change = (latest - prior).sort_values()
        print("\nQ-o-Q change in win rate (latest vs prior quarter):")
        print(change.round(3).to_string())
    print("\n--- Why it matters: Identifies which regions are dragging overall win rate.")
    print("--- Action: Focus coaching and process in regions with largest drop or lowest rate.")
    if HAS_PLOT:
        pivot.plot(kind="line", marker="o", figsize=(10, 4), title="Win rate by region over time")
        plt.legend(bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "insight1_win_rate_region_quarter.png", dpi=120, bbox_inches="tight")
        plt.close()
    return wr


def insight_2_lead_source_and_product_impact(df: pd.DataFrame):
    """
    Insight 2: Win rate by lead source and product type — what to double down on.
    """
    wr_source = df.groupby("lead_source").agg(wins=("won", "sum"), total=("won", "count")).assign(
        win_rate=lambda x: x["wins"] / x["total"]
    )
    wr_product = df.groupby("product_type").agg(wins=("won", "sum"), total=("won", "count")).assign(
        win_rate=lambda x: x["wins"] / x["total"]
    )
    print("\n" + "=" * 60)
    print("INSIGHT 2: Win rate by lead source and product type")
    print("=" * 60)
    print("By lead_source:")
    print(wr_source[["wins", "total", "win_rate"]].round(3).to_string())
    print("\nBy product_type:")
    print(wr_product[["wins", "total", "win_rate"]].round(3).to_string())
    print("\n--- Why it matters: Shows which channels and products convert; mix shift can explain overall drop.")
    print("--- Action: Invest in high win-rate sources/products; fix or qualify better in low win-rate ones.")
    return wr_source, wr_product


def insight_3_sales_cycle_and_outcome(df: pd.DataFrame):
    """
    Insight 3: Sales cycle length by outcome — long cycles and losses.
    """
    by_outcome = df.groupby("outcome")["sales_cycle_days"].agg(["median", "mean", "count"])
    print("\n" + "=" * 60)
    print("INSIGHT 3: Sales cycle length by outcome")
    print("=" * 60)
    print(by_outcome.to_string())
    # Correlation: longer cycle -> more losses?
    corr = df[["sales_cycle_days", "won"]].corr().loc["sales_cycle_days", "won"]
    print(f"\nCorrelation(sales_cycle_days, won): {corr:.3f}")
    print("\n--- Why it matters: Long cycles often indicate stalls, objections, or poor fit.")
    print("--- Action: Flag deals aging past segment median; review stage progression and qualification.")
    return by_outcome


def print_custom_metrics(df: pd.DataFrame):
    """Compute and print the two custom metrics with business explanation."""
    pqs = custom_metric_pipeline_quality_score(df)
    cwe = custom_metric_cycle_win_efficiency(df)
    print("\n" + "=" * 60)
    print("CUSTOM METRIC 1: Pipeline Quality Score (PQS)")
    print("=" * 60)
    print("Segment-level historical win rate (= PQS for that segment).")
    print(pqs.nlargest(15, "pqs").to_string())
    print("\n--- Definition: PQS(segment) = historical win rate in that segment.")
    print("--- Use: Weight pipeline by PQS to see 'quality-adjusted' pipeline; low-PQS segments need qualification or process.")
    print("\n" + "=" * 60)
    print("CUSTOM METRIC 2: Cycle Win Efficiency (CWE)")
    print("=" * 60)
    print("Win rate / (segment_median_cycle / global_median_cycle). Higher = efficient wins.")
    print(cwe.sort_values("cwe", ascending=False).to_string())
    print("\n--- Definition: CWE = win_rate / relative_cycle_length. High CWE = win without dragging cycle.")
    print("--- Use: Prioritize segments with low CWE for process/coaching (long cycles or low win rate).")
    return pqs, cwe


def main():
    df = load_data()
    run_eda(df)
    insight_1_win_rate_by_region_and_quarter(df)
    insight_2_lead_source_and_product_impact(df)
    insight_3_sales_cycle_and_outcome(df)
    print_custom_metrics(df)
    print("\nDone. Optional plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
