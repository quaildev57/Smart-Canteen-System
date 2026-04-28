import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

df = pd.read_csv("canteen_clean.csv", parse_dates=["Date"])
os.makedirs("charts", exist_ok=True)

BG      = "#F7F9FC"
PRIMARY = "#1B4F72"
ACCENT  = "#2ECC71"
WARN    = "#E74C3C"
PALETTE = ["#1B4F72","#2ECC71","#F39C12","#E74C3C","#9B59B6",
           "#1ABC9C","#E67E22","#3498DB","#D35400","#27AE60"]

def save(name):
    plt.savefig(f"charts/{name}.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close()
    print(f"  ✅ charts/{name}.png")

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.labelsize": 11,
})

print("Generating charts...\n")


#daily demand
daily = df.groupby("Date")["Quantity"].sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 4), facecolor=BG)
ax.fill_between(daily["Date"], daily["Quantity"], alpha=0.18, color=PRIMARY)
ax.plot(daily["Date"], daily["Quantity"], color=PRIMARY, linewidth=2.2, marker="o", markersize=3)
ax.set_title("Daily Total Orders Over Time")
ax.set_xlabel("Date"); ax.set_ylabel("Total Quantity Ordered")
ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
plt.xticks(rotation=30)
save("1_daily_trend")

#top items
top_items = df.groupby("Item")["Quantity"].sum().sort_values(ascending=True).tail(10)
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
bars = ax.barh(top_items.index, top_items.values, color=PALETTE[:10], edgecolor="white")
for bar, val in zip(bars, top_items.values):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=10, color="#333")
ax.set_title("Top 10 Most Ordered Items")
ax.set_xlabel("Total Quantity Ordered")
save("2_top_items")

# peak hours
hourly = df.groupby("Hour")["Quantity"].sum().reset_index()
fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
colors = [WARN if h in [8,9,12,13,16,17] else PRIMARY for h in hourly["Hour"]]
ax.bar(hourly["Hour"], hourly["Quantity"], color=colors, edgecolor="white", width=0.7)
ax.set_title("Peak Order Hours  (red = rush hours)")
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Total Quantity")
ax.set_xticks(hourly["Hour"])
save("3_peak_hours")

# weather vs quantity
weather_avg = df.groupby("Weather")["Quantity"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
bars = ax.bar(weather_avg.index, weather_avg.values,
              color=[PRIMARY, ACCENT, WARN, "#F39C12"][:len(weather_avg)],
              edgecolor="white", width=0.5)
for bar, val in zip(bars, weather_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")
ax.set_title("Avg Quantity Ordered by Weather")
ax.set_xlabel("Weather"); ax.set_ylabel("Avg Quantity")
save("4_weather_demand")

# day of week demand
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
day_data  = df.groupby("DayOfWeek")["Quantity"].sum().reindex(day_order).dropna()
fig, ax   = plt.subplots(figsize=(9, 4), facecolor=BG)
ax.bar(day_data.index, day_data.values, color=PRIMARY, edgecolor="white", width=0.55)
for i, val in enumerate(day_data.values):
    ax.text(i, val + 2, str(int(val)), ha="center", fontsize=10)
ax.set_title("Total Orders by Day of Week")
ax.set_xlabel("Day"); ax.set_ylabel("Total Quantity")
save("5_day_demand")

# meal time of day
meal_counts = df.groupby("Meal_Time")["Quantity"].sum()
fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
wedges, texts, autotexts = ax.pie(
    meal_counts.values, labels=meal_counts.index,
    autopct="%1.1f%%", colors=[PRIMARY, ACCENT, "#F39C12"],
    startangle=140, wedgeprops=dict(edgecolor="white", linewidth=2))
for t in autotexts: t.set_fontsize(12); t.set_fontweight("bold")
ax.set_title("Order Share by Meal Time")
save("6_meal_time_split")

# rating
fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
ax.hist(df["Rating"], bins=20, color=PRIMARY, edgecolor="white", alpha=0.85)
ax.axvline(df["Rating"].mean(), color=WARN, linestyle="--", linewidth=2,
           label=f"Mean: {df['Rating'].mean():.2f}")
ax.set_title("Customer Rating Distribution")
ax.set_xlabel("Rating"); ax.set_ylabel("Count")
ax.legend()
save("7_rating_dist")

# payment mode
pay = df["Payment_Mode"].value_counts()
fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
ax.pie(pay.values, labels=pay.index, autopct="%1.1f%%",
       colors=[PRIMARY, ACCENT, "#F39C12"],
       startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
ax.set_title("Payment Mode Distribution")
save("8_payment_mode")

# heatmap of correlations
cols = ["Quantity","Total_Amount","Rating","Hour",
        "DayOfWeek_enc","Meal_Time_enc","Weather_enc","Wastage_Risk"]
corr = df[cols].corr()
fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap")
plt.xticks(rotation=30); plt.yticks(rotation=0)
save("9_heatmap")

# top items by avg rating
item_rating = df.groupby("Item")["Rating"].mean().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
colors_r = [ACCENT if v >= 4.0 else WARN for v in item_rating.values]
ax.barh(item_rating.index, item_rating.values, color=colors_r, edgecolor="white")
ax.axvline(4.0, color="#333", linestyle="--", linewidth=1.2, label="Rating = 4.0")
ax.set_title("Avg Customer Rating per Item  (green ≥ 4.0)")
ax.set_xlabel("Average Rating")
ax.legend()
save("10_item_ratings")

print("\n All 10 charts saved to charts/ folder!")
