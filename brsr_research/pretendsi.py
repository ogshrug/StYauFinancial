import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

OUTPUT_FOLDER = r"./output"

print("Loading spreads...")
df = pd.read_parquet(f"{OUTPUT_FOLDER}/spreads.parquet")

df["LOG_SPREAD"] = np.log(df["CS_SPREAD"].replace(0, np.nan))
df = df.dropna(subset=["LOG_SPREAD"])

# ── Event study: quarters relative to Apr 2022 announcement ──────────────────
df["QUARTER"] = pd.PeriodIndex(df["DATE"], freq="Q")
base = pd.Period("2022Q1", freq="Q")  # quarter just before announcement
df["REL_Q"] = (df["QUARTER"] - base).apply(lambda x: x.n)

# Keep -4 to +4 quarters around announcement
df_es = df[(df["REL_Q"] >= -4) & (df["REL_Q"] <= 4)].copy()

# Create dummies for each quarter (omit -1 as baseline)
for q in range(-4, 5):
    if q != -1:
        df_es[f"Q{q}"] = ((df_es["REL_Q"] == q) & (df_es["TREATED"] == 1)).astype(int)

q_vars = [f"Q{q}" for q in range(-4, 5) if q != -1]
formula = "LOG_SPREAD ~ " + " + ".join(q_vars) + " + C(REL_Q) + TREATED"

# Demean by firm
vars_needed = ["LOG_SPREAD", "TREATED"] + q_vars + ["REL_Q", "SYMBOL"]
df_es2 = df_es[vars_needed].copy()
demean_vars = ["LOG_SPREAD"] + q_vars
firm_means = df_es2.groupby("SYMBOL")[demean_vars].transform("mean")
for v in demean_vars:
    df_es2[v] = df_es2[v] - firm_means[v]

es_model = smf.ols(formula, data=df_es2).fit(cov_type="HC3")

# ── Plot event study ──────────────────────────────────────────────────────────
quarters = [q for q in range(-4, 5) if q != -1]
coefs    = [es_model.params.get(f"Q{q}", 0) for q in quarters]
cis      = [es_model.conf_int().loc[f"Q{q}"].values if f"Q{q}" in es_model.params else [0,0] for q in quarters]

# Insert 0 for omitted quarter -1
all_quarters = list(range(-4, 5))
all_coefs    = []
all_ci_low   = []
all_ci_high  = []
for q in all_quarters:
    if q == -1:
        all_coefs.append(0)
        all_ci_low.append(0)
        all_ci_high.append(0)
    else:
        idx = quarters.index(q)
        all_coefs.append(coefs[idx])
        all_ci_low.append(cis[idx][0])
        all_ci_high.append(cis[idx][1])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(all_quarters, all_coefs, "o-", color="#1f77b4", linewidth=2, label="DiD estimate")
ax.fill_between(all_quarters, all_ci_low, all_ci_high, alpha=0.2, color="#1f77b4", label="95% CI")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axvline(-0.5, color="red", linewidth=1.5, linestyle=":", label="BRSR Announced (Apr 2022)")
ax.set_xlabel("Quarters Relative to BRSR Announcement", fontsize=12)
ax.set_ylabel("DiD Coefficient (Log Spread)", fontsize=12)
ax.set_title("Event Study: Pre-trends Test\nTreatment vs Control Bid-Ask Spreads", fontsize=13, fontweight="bold")
ax.set_xticks(all_quarters)
ax.set_xticklabels([f"Q{q:+d}" for q in all_quarters])
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/pretrends.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved pretrends.png")

# ── Pre-trend test: are pre-announcement coefficients jointly zero? ───────────
pre_vars = [f"Q{q}" for q in range(-4, -1)]
from statsmodels.stats.wald_test import WaldTestResults
wald = es_model.f_test([f"Q{q} = 0" for q in range(-4, -1)])
print(f"\nPre-trends Wald test (H0: no pre-trends):")
print(f"  F-stat  : {wald.statistic[0][0]:.4f}")
print(f"  P-value : {wald.pvalue:.4f}")
print("\nIf p > 0.05, parallel pre-trends assumption holds.")