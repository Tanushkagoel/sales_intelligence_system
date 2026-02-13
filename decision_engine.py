"""
Part 3 – Decision Engine: Win Rate Driver Analysis (Option B)
Scores which factors hurt or improve win rate and produces actionable outputs.
Run: python decision_engine.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "skygeni_sales_data.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# --- Problem definition ---
# We want to answer: "Which factors are hurting or improving win rate?"
# Output: Ranked drivers (region, industry, product, lead_source, deal_stage, amount band, cycle)
# and segment-level recommendations (e.g. "APAC has -X% lift; consider regional playbook").


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df["created_date"] = pd.to_datetime(df["created_date"])
    df["closed_date"] = pd.to_datetime(df["closed_date"])
    df["won"] = (df["outcome"] == "Won").astype(int)
    # Bands for interpretability
    df["amount_band"] = pd.cut(
        df["deal_amount"],
        bins=[0, 10_000, 25_000, 50_000, 100_000, np.inf],
        labels=["<10k", "10-25k", "25-50k", "50-100k", "100k+"],
    )
    df["cycle_band"] = pd.cut(
        df["sales_cycle_days"],
        bins=[0, 30, 60, 90, 120, np.inf],
        labels=["0-30d", "31-60d", "61-90d", "91-120d", "120d+"],
    )
    return df


def encode_features(df: pd.DataFrame, categorical_cols: list, fit_encoders: dict = None):
    """Encode categoricals; return X, encoders (for reuse on new data)."""
    encoders = fit_encoders or {}
    X = df.copy()
    for col in categorical_cols:
        if col not in X.columns:
            continue
        if fit_encoders is None or col not in encoders:
            encoders[col] = LabelEncoder()
            encoders[col].fit(X[col].astype(str).fillna("NA"))
        X[col] = encoders[col].transform(X[col].astype(str).fillna("NA"))
    return X, encoders


def get_driver_importance(df: pd.DataFrame):
    """
    Simple model: logistic regression. Coefficients + segment-level win rate deltas
    give us 'drivers'. We also compute segment lift vs baseline for plain-language output.
    """
    categorical_cols = ["region", "industry", "product_type", "lead_source", "deal_stage", "amount_band", "cycle_band"]
    df_enc, encoders = encode_features(df, categorical_cols)
    feature_cols = [c for c in categorical_cols if c in df_enc.columns]
    X = df_enc[feature_cols]
    y = df_enc["won"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    # Coefficient interpretation: positive = associated with win
    coef = pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_[0]})
    coef["abs_effect"] = np.abs(coef["coefficient"])
    coef = coef.sort_values("abs_effect", ascending=False)
    return model, encoders, coef, feature_cols, X_train, y_train


def segment_lift_table(df: pd.DataFrame):
    """For each segment dimension, show lift vs overall baseline (so leaders see 'where we win/lose')."""
    baseline = df["won"].mean()
    dimensions = ["region", "industry", "product_type", "lead_source", "deal_stage"]
    rows = []
    for dim in dimensions:
        g = df.groupby(dim).agg(wins=("won", "sum"), total=("won", "count")).assign(
            win_rate=lambda x: x["wins"] / x["total"],
            lift=lambda x: (x["win_rate"] - baseline) * 100,
        )
        g = g.reset_index()
        g["dimension"] = dim
        g["segment"] = g[dim]  # single column for display
        rows.append(g[["dimension", "segment", "wins", "total", "win_rate", "lift"]])
    lift_df = pd.concat(rows, ignore_index=True)
    return lift_df, baseline


def generate_actionable_outputs(df: pd.DataFrame, coef: pd.DataFrame, lift_df: pd.DataFrame, baseline: float):
    """Produce text and CSV outputs a sales leader can use."""
    lines = []
    lines.append("=" * 60)
    lines.append("WIN RATE DRIVER ANALYSIS – Actionable Summary")
    lines.append("=" * 60)
    lines.append(f"\nOverall win rate (baseline): {baseline:.1%}")
    lines.append("\n--- Top factors associated with WIN (positive drivers) ---")
    positive = coef[coef["coefficient"] > 0].head(5)
    for _, r in positive.iterrows():
        lines.append(f"  {r['feature']}: coefficient {r['coefficient']:.3f}")
    lines.append("\n--- Top factors associated with LOSS (negative drivers) ---")
    negative = coef[coef["coefficient"] < 0].head(5)
    for _, r in negative.iterrows():
        lines.append(f"  {r['feature']}: coefficient {r['coefficient']:.3f}")

    lines.append("\n--- Segment lift vs baseline (lift = segment_win_rate - baseline, in pp) ---")
    lines.append("Segments with largest NEGATIVE lift (hurting win rate):")
    worst_display = lift_df.nsmallest(10, "lift")
    lines.append(worst_display.to_string())
    lines.append("\nSegments with largest POSITIVE lift (helping win rate):")
    best_display = lift_df.nlargest(10, "lift")
    lines.append(best_display.to_string())

    lines.append("\n--- How a sales leader would use this ---")
    lines.append("1. Focus on segments with largest negative lift: coaching, playbooks, or qualification.")
    lines.append("2. Double down on segments with positive lift: replicate what works (source, product, region).")
    lines.append("3. Use coefficient signs to prioritize: fix negative drivers before scaling positive ones.")
    lines.append("4. Re-run monthly/quarterly to track whether interventions improve segment win rates.")
    report = "\n".join(lines)
    print(report)
    return report


def main():
    df = load_and_prepare()
    # Drop rows with missing key categoricals for model
    model_df = df.dropna(subset=["region", "industry", "product_type", "lead_source", "deal_stage"])
    model_df = model_df[model_df["amount_band"].notna() & model_df["cycle_band"].notna()]
    model, encoders, coef, feature_cols, X_train, y_train = get_driver_importance(model_df)
    lift_df, baseline = segment_lift_table(df)
    report = generate_actionable_outputs(df, coef, lift_df, baseline)
    # Save outputs
    (OUTPUT_DIR / "win_rate_drivers_coefficients.csv").write_text(coef.to_csv(index=False))
    lift_df.to_csv(OUTPUT_DIR / "win_rate_segment_lift.csv", index=False)
    (OUTPUT_DIR / "win_rate_driver_report.txt").write_text(report)
    print("\nOutputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
