import pandas as pd
import numpy as np

OUTPUT_FOLDER = r"./output"

print("Loading sample...")
df = pd.read_parquet(f"{OUTPUT_FOLDER}/sample.parquet")
print(f"Loaded {len(df):,} rows\n")

# ── Corwin-Schultz (2012) bid-ask spread estimator ───────────────────────────
def corwin_schultz(df):
    df = df.sort_values("DATE").copy()

    log_hl  = np.log(df["HIGH"] / df["LOW"])
    log_hl2 = log_hl ** 2

    # Two-day high and low using shift (safe inside groupby)
    two_day_high = np.log(df[["HIGH"]].join(df["HIGH"].shift().rename("HIGH_PREV")).max(axis=1))
    two_day_low  = np.log(df[["LOW"]].join(df["LOW"].shift().rename("LOW_PREV")).min(axis=1))

    beta  = log_hl2 + log_hl2.shift(1)
    gamma = (two_day_high - two_day_low) ** 2

    k     = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)

    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    # Negative spreads set to zero (standard in literature)
    spread = spread.clip(lower=0)

    df["CS_SPREAD"] = spread
    return df

print("Calculating Corwin-Schultz spreads...")
print("(This may take 2-3 minutes)\n")

result = (
    df.groupby("SYMBOL", group_keys=False)
      .apply(corwin_schultz)
)

result = result.dropna(subset=["CS_SPREAD"])

print(f"Rows after spread calculation : {len(result):,}")
print(f"Mean spread (all)             : {result['CS_SPREAD'].mean():.4f}")
print(f"Mean spread (treatment)       : {result[result['TREATED']==1]['CS_SPREAD'].mean():.4f}")
print(f"Mean spread (control)         : {result[result['TREATED']==0]['CS_SPREAD'].mean():.4f}")
print(f"Spread = 0 (pct)              : {(result['CS_SPREAD']==0).mean()*100:.1f}%")

result.to_parquet(f"{OUTPUT_FOLDER}/spreads.parquet", index=False)
print(f"\nSaved spreads.parquet to {OUTPUT_FOLDER}")