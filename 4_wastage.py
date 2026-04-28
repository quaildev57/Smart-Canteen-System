
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#Load Data 
df = pd.read_csv("canteen_clean.csv", parse_dates=["Date"])

BG      = "#F7F9FC"
PRIMARY = "#1B4F72"
ACCENT  = "#2ECC71"
WARN    = "#E74C3C"
ORANGE  = "#F39C12"
PURPLE  = "#9B59B6"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelsize": 11,
})

def save(name):
    plt.savefig(f"charts/{name}.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   charts/{name}.png")

# wastage risk
df["Wastage_Risk"] = ((df["Rating"] < 3.5) & (df["Quantity"] >= 2)).astype(int)

df["Waste_Score"] = np.where(df["Rating"] < 3.5, df["Quantity"], 0)

print("=" * 55)
print("  STEP 4 — WASTAGE ANALYSIS")
print("=" * 55)
print(f"  Total transactions     : {len(df):,}")
print(f"  High wastage risk rows : {df['Wastage_Risk'].sum()} "
      f"({df['Wastage_Risk'].mean()*100:.1f}%)")
print(f"  Avg rating overall     : {df['Rating'].mean():.2f}")
print(f"  Rows with rating < 3.5 : {(df['Rating'] < 3.5).sum()}")
print("=" * 55)

# Chart 1: Wastage Risk by Item 
item_waste = (df.groupby("Item")["Wastage_Risk"]
                .agg(["sum","count"])
                .rename(columns={"sum":"waste_count","count":"total"}))
item_waste["waste_pct"] = item_waste["waste_count"] / item_waste["total"] * 100
item_waste = item_waste.sort_values("waste_pct", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
colors = [WARN if v >= 4 else ORANGE if v >= 2 else ACCENT
          for v in item_waste["waste_pct"]]
bars = ax.barh(item_waste.index, item_waste["waste_pct"],
               color=colors, edgecolor="white")
for bar, val in zip(bars, item_waste["waste_pct"]):
    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9)
ax.axvline(item_waste["waste_pct"].mean(), color=PRIMARY,
           linestyle="--", linewidth=1.5,
           label=f"Avg: {item_waste['waste_pct'].mean():.1f}%")
ax.set_title("Wastage Risk % by Food Item")
ax.set_xlabel("Wastage Risk %")
patches = [mpatches.Patch(color=WARN,   label="High Risk (≥4%)"),
           mpatches.Patch(color=ORANGE, label="Medium Risk (2–4%)"),
           mpatches.Patch(color=ACCENT, label="Low Risk (<2%)")]
ax.legend(handles=patches + [plt.Line2D([0],[0], color=PRIMARY,
          linestyle="--", label=f"Avg {item_waste['waste_pct'].mean():.1f}%")],
          loc="lower right", fontsize=9)
save("14_wastage_by_item")

# Chart 2: Wastage Risk by Day of Week
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
day_waste = (df.groupby("DayOfWeek")["Wastage_Risk"]
               .agg(["sum","count"])
               .rename(columns={"sum":"waste_count","count":"total"}))
day_waste["waste_pct"] = day_waste["waste_count"] / day_waste["total"] * 100
day_waste = day_waste.reindex(day_order).dropna()

fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
bar_colors = [WARN if v >= 4 else ORANGE if v >= 2 else ACCENT
              for v in day_waste["waste_pct"]]
bars = ax.bar(day_waste.index, day_waste["waste_pct"],
              color=bar_colors, edgecolor="white", width=0.55)
for bar, val in zip(bars, day_waste["waste_pct"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1, f"{val:.1f}%",
            ha="center", fontsize=10, fontweight="bold")
ax.axhline(day_waste["waste_pct"].mean(), color=PRIMARY,
           linestyle="--", linewidth=1.5,
           label=f"Avg: {day_waste['waste_pct'].mean():.1f}%")
ax.set_title("Wastage Risk % by Day of Week")
ax.set_xlabel("Day"); ax.set_ylabel("Wastage Risk %")
ax.legend()
save("15_wastage_by_day")

# Chart 3: Wastage Risk by Meal Time
meal_waste = (df.groupby("Meal_Time")["Wastage_Risk"]
                .agg(["sum","count"])
                .rename(columns={"sum":"waste_count","count":"total"}))
meal_waste["waste_pct"] = meal_waste["waste_count"] / meal_waste["total"] * 100

fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
meal_colors = [WARN if v >= 4 else ORANGE if v >= 2 else ACCENT
               for v in meal_waste["waste_pct"]]
bars = ax.bar(meal_waste.index, meal_waste["waste_pct"],
              color=meal_colors, edgecolor="white", width=0.45)
for bar, val in zip(bars, meal_waste["waste_pct"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1, f"{val:.1f}%",
            ha="center", fontsize=11, fontweight="bold")
ax.set_title("Wastage Risk % by Meal Time")
ax.set_xlabel("Meal Time"); ax.set_ylabel("Wastage Risk %")
save("16_wastage_by_meal")

# Chart 4: Wastage Risk by Weather
weather_waste = (df.groupby("Weather")["Wastage_Risk"]
                   .agg(["sum","count"])
                   .rename(columns={"sum":"waste_count","count":"total"}))
weather_waste["waste_pct"] = (weather_waste["waste_count"]
                               / weather_waste["total"] * 100)
weather_waste = weather_waste.sort_values("waste_pct", ascending=False)

fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
w_colors = [WARN if v >= 4 else ORANGE if v >= 2 else ACCENT
            for v in weather_waste["waste_pct"]]
bars = ax.bar(weather_waste.index, weather_waste["waste_pct"],
              color=w_colors, edgecolor="white", width=0.45)
for bar, val in zip(bars, weather_waste["waste_pct"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1, f"{val:.1f}%",
            ha="center", fontsize=11, fontweight="bold")
ax.set_title("Wastage Risk % by Weather Condition")
ax.set_xlabel("Weather"); ax.set_ylabel("Wastage Risk %")
save("17_wastage_by_weather")

# Chart 5: Rating vs Waste Score Scatter
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
scatter_colors = [WARN if w > 0 else PRIMARY for w in df["Waste_Score"]]
ax.scatter(df["Rating"], df["Quantity"],
           c=scatter_colors, alpha=0.35, s=25, edgecolors="none")
ax.axvline(3.5, color=ORANGE, linestyle="--", linewidth=2,
           label="Risk Threshold (3.5)")
ax.set_title("Rating vs Quantity  (red = wastage risk)")
ax.set_xlabel("Customer Rating")
ax.set_ylabel("Quantity Ordered")
ax.legend()
risk_patch   = mpatches.Patch(color=WARN,    label="Wastage Risk")
norisk_patch = mpatches.Patch(color=PRIMARY, label="No Risk")
ax.legend(handles=[risk_patch, norisk_patch,
          plt.Line2D([0],[0], color=ORANGE, linestyle="--",
                     label="Threshold (3.5)")])
save("18_rating_vs_waste")

# ── Chart 6: Top 5 High-Risk Items — Detail ───────────────────
top5 = item_waste.sort_values("waste_pct", ascending=False).head(5)
avg_ratings = df[df["Item"].isin(top5.index)].groupby("Item")["Rating"].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)

ax1.barh(top5.index, top5["waste_pct"], color=WARN, edgecolor="white")
for i, val in enumerate(top5["waste_pct"]):
    ax1.text(val + 0.05, i, f"{val:.1f}%", va="center", fontsize=10)
ax1.set_title("Top 5 High-Risk Items — Wastage %")
ax1.set_xlabel("Wastage Risk %")
ax1.set_facecolor(BG)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

avg_r = avg_ratings.reindex(top5.index)
bar_cols = [WARN if v < 3.8 else ORANGE for v in avg_r.values]
ax2.barh(avg_r.index, avg_r.values, color=bar_cols, edgecolor="white")
ax2.axvline(3.5, color=PRIMARY, linestyle="--", linewidth=1.5,
            label="Risk threshold 3.5")
for i, val in enumerate(avg_r.values):
    ax2.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=10)
ax2.set_title("Avg Rating — Same Top 5 Items")
ax2.set_xlabel("Average Rating")
ax2.legend(fontsize=9)
ax2.set_facecolor(BG)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

plt.tight_layout()
save("19_top5_risk_items")

# ── Wastage Summary Table ─────────────────────────────────────
print("\n WASTAGE RISK SUMMARY TABLE")
print("-" * 55)
summary = item_waste.sort_values("waste_pct", ascending=False)[
    ["waste_count","total","waste_pct"]].copy()
summary.columns = ["Risk Txns", "Total Txns", "Risk %"]
summary["Avg Rating"] = df.groupby("Item")["Rating"].mean().round(2)
summary["Risk Level"] = summary["Risk %"].apply(
    lambda x: "🔴 High" if x >= 4 else ("🟡 Medium" if x >= 2 else "🟢 Low"))
print(summary.to_string())

print("\n WASTAGE BY DAY")
print("-" * 40)
print(day_waste[["waste_count","total","waste_pct"]].rename(
    columns={"waste_count":"Risk Txns","total":"Total","waste_pct":"Risk %"}
).round(2).to_string())

print("\n WASTAGE BY MEAL TIME")
print("-" * 40)
print(meal_waste[["waste_count","total","waste_pct"]].rename(
    columns={"waste_count":"Risk Txns","total":"Total","waste_pct":"Risk %"}
).round(2).to_string())

print("\n Step 4 Complete! All wastage charts saved to charts/")
