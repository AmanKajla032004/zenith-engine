"""Zenith v1.2 — CLI entry point with labeled graphs"""

import matplotlib.pyplot as plt

from zenith import analyze, generate_report
from zenith.pipeline import load_data
from zenith.config import ZenithConfig
from zenith.scoring import compute_domain_scores
from zenith.signals import compute_rolling_means, compute_volatility


if __name__ == "__main__":

    DATA_FILE = "test_data.json"

    # ---- Run analysis (text report) ----
    result = analyze(DATA_FILE)
    print(generate_report(result))

    # ---- Prepare dataframe for plotting ----
    cfg = ZenithConfig()
    df = load_data(DATA_FILE)
    df = compute_domain_scores(df, cfg)
    df = compute_rolling_means(df, cfg)
    df = compute_volatility(df, cfg)

    # -----------------------------------------
    # 1️⃣ Global Balance + Rolling Means
    # -----------------------------------------
    plt.figure()
    plt.plot(df["date"], df["global_balance"], label="Global Balance")
    plt.plot(df["date"], df["global_balance_7d_mean"], label="7-Day Mean")
    plt.plot(df["date"], df["global_balance_30d_mean"], label="30-Day Mean")
    plt.title("Global Balance & Rolling Means")
    plt.xlabel("Date")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------------------
    # 2️⃣ Domain Scores
    # -----------------------------------------
    plt.figure()
    plt.plot(df["date"], df["recovery_score"], label="Recovery")
    plt.plot(df["date"], df["emotional_score"], label="Emotional")
    plt.plot(df["date"], df["performance_score"], label="Performance")
    plt.plot(df["date"], df["energy_focus_score"], label="Energy/Focus")
    plt.title("Domain Scores Over Time")
    plt.xlabel("Date")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------------------
    # 3️⃣ Volatility
    # -----------------------------------------
    plt.figure()
    plt.plot(df["date"], df["volatility_index"], label="Volatility Index")
    plt.title("Volatility Index (7-Day Rolling)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()