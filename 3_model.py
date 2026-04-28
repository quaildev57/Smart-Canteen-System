import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

#Load Data
df = pd.read_csv("canteen_clean.csv", parse_dates=["Date"])

BG      = "#F7F9FC"
PRIMARY = "#1B4F72"
ACCENT  = "#2ECC71"
WARN    = "#E74C3C"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

# Feature Engineering
# Encode any remaining string columns
le_item    = LabelEncoder()
le_weather = LabelEncoder()
le_meal    = LabelEncoder()
le_day     = LabelEncoder()

df["Item_enc"]       = le_item.fit_transform(df["Item"])
df["Weather_enc"]    = le_weather.fit_transform(df["Weather"])
df["Meal_Time_enc"]  = le_meal.fit_transform(df["Meal_Time"])
df["DayOfWeek_enc"]  = le_day.fit_transform(df["DayOfWeek"])

# Features used for prediction
FEATURES = ["DayOfWeek_enc", "Meal_Time_enc", "Weather_enc",
            "Hour", "Item_enc", "Actual_Item_Price"]
TARGET   = "Quantity"

X = df[FEATURES]
y = df[TARGET]

# Train / Test Split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("=" * 55)
print("  STEP 3 — ML DEMAND PREDICTION")
print("=" * 55)
print(f"  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")
print(f"  Features         : {FEATURES}")
print(f"  Target           : {TARGET}")
print("=" * 55)

# Model 1: Random Forest
print("\n🌲 Training Random Forest...")
rf = RandomForestRegressor(n_estimators=200, max_depth=8,
                           random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_pred_r = np.round(rf_pred).astype(int).clip(1, 3)

rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2   = r2_score(y_test, rf_pred)
rf_acc  = (rf_pred_r == y_test.values).mean() * 100

print(f"   MAE  : {rf_mae:.4f}")
print(f"   RMSE : {rf_rmse:.4f}")
print(f"   R²   : {rf_r2:.4f}")
print(f"   Exact Accuracy : {rf_acc:.1f}%")

#Model 2: XGBoost
print("\n⚡ Training XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6,
                               learning_rate=0.1, random_state=42,
                               verbosity=0)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_pred_r = np.round(xgb_pred).astype(int).clip(1, 3)

xgb_mae  = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2   = r2_score(y_test, xgb_pred)
xgb_acc  = (xgb_pred_r == y_test.values).mean() * 100

print(f"   MAE  : {xgb_mae:.4f}")
print(f"   RMSE : {xgb_rmse:.4f}")
print(f"   R²   : {xgb_r2:.4f}")
print(f"   Exact Accuracy : {xgb_acc:.1f}%")

#Results Summary
print("\n" + "=" * 55)
print("  MODEL COMPARISON SUMMARY")
print("=" * 55)
print(f"  {'Metric':<20} {'Random Forest':>15} {'XGBoost':>12}")
print(f"  {'-'*47}")
print(f"  {'MAE':<20} {rf_mae:>15.4f} {xgb_mae:>12.4f}")
print(f"  {'RMSE':<20} {rf_rmse:>15.4f} {xgb_rmse:>12.4f}")
print(f"  {'R² Score':<20} {rf_r2:>15.4f} {xgb_r2:>12.4f}")
print(f"  {'Exact Accuracy':<20} {rf_acc:>14.1f}% {xgb_acc:>11.1f}%")
winner = "Random Forest" if rf_mae < xgb_mae else "XGBoost"
print(f"\n  🏆 Better Model (lower MAE): {winner}")
print("=" * 55)

#Chart 1: Model Comparison Bar Chart
metrics = ["MAE", "RMSE", "R² Score"]
rf_vals  = [rf_mae,  rf_rmse,  rf_r2]
xgb_vals = [xgb_mae, xgb_rmse, xgb_r2]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=BG)
fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", y=1.02)

for i, (ax, metric, rv, xv) in enumerate(zip(axes, metrics, rf_vals, xgb_vals)):
    bars = ax.bar(["Random\nForest", "XGBoost"], [rv, xv],
                  color=[PRIMARY, ACCENT], edgecolor="white", width=0.45)
    for bar, val in zip(bars, [rv, xv]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title(metric)
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("charts/11_model_comparison.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("\n  ✅ charts/11_model_comparison.png")

#Chart 2: Feature Importance (both models)
feat_labels = ["Day of Week", "Meal Time", "Weather",
               "Hour", "Item", "Item Price"]

rf_imp  = rf.feature_importances_
xgb_imp = xgb_model.feature_importances_

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)

# RF importance
idx1 = np.argsort(rf_imp)
ax1.barh([feat_labels[i] for i in idx1], rf_imp[idx1],
         color=PRIMARY, edgecolor="white")
ax1.set_title("🌲 Random Forest — Feature Importance")
ax1.set_xlabel("Importance Score")
ax1.set_facecolor(BG); ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# XGBoost importance
idx2 = np.argsort(xgb_imp)
ax2.barh([feat_labels[i] for i in idx2], xgb_imp[idx2],
         color=ACCENT, edgecolor="white")
ax2.set_title("⚡ XGBoost — Feature Importance")
ax2.set_xlabel("Importance Score")
ax2.set_facecolor(BG); ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("charts/12_feature_importance.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("  ✅ charts/12_feature_importance.png")

#Chart 3: Actual vs Predicted
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)

sample = min(80, len(y_test))
idx    = np.arange(sample)

ax1.plot(idx, y_test.values[:sample], "o-", color=PRIMARY,
         label="Actual", markersize=4, linewidth=1.5)
ax1.plot(idx, rf_pred_r[:sample], "s--", color=WARN,
         label="Predicted", markersize=4, linewidth=1.5)
ax1.set_title("🌲 Random Forest — Actual vs Predicted")
ax1.set_xlabel("Sample Index"); ax1.set_ylabel("Quantity")
ax1.legend(); ax1.set_facecolor(BG)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

ax2.plot(idx, y_test.values[:sample], "o-", color=PRIMARY,
         label="Actual", markersize=4, linewidth=1.5)
ax2.plot(idx, xgb_pred_r[:sample], "s--", color=ACCENT,
         label="Predicted", markersize=4, linewidth=1.5)
ax2.set_title("⚡ XGBoost — Actual vs Predicted")
ax2.set_xlabel("Sample Index"); ax2.set_ylabel("Quantity")
ax2.legend(); ax2.set_facecolor(BG)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("charts/13_actual_vs_predicted.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("  ✅ charts/13_actual_vs_predicted.png")

#Live Prediction Demo
print("\n" + "=" * 55)
print("  LIVE PREDICTION DEMO")
print("=" * 55)

test_cases = [
    {"DayOfWeek": "Monday",    "Meal_Time": "Lunch",
     "Weather": "Sunny",  "Hour": 12, "Item": "Dosa",    "Price": 50},
    {"DayOfWeek": "Thursday",  "Meal_Time": "Breakfast",
     "Weather": "Rainy",  "Hour": 9,  "Item": "Tea",     "Price": 15},
    {"DayOfWeek": "Saturday",  "Meal_Time": "Evening Snacks",
     "Weather": "Cloudy", "Hour": 17, "Item": "Sandwich","Price": 40},
]

for i, case in enumerate(test_cases, 1):
    row = pd.DataFrame([{
        "DayOfWeek_enc"   : le_day.transform([case["DayOfWeek"]])[0],
        "Meal_Time_enc"   : le_meal.transform([case["Meal_Time"]])[0],
        "Weather_enc"     : le_weather.transform([case["Weather"]])[0],
        "Hour"            : case["Hour"],
        "Item_enc"        : le_item.transform([case["Item"]])[0],
        "Actual_Item_Price": case["Price"],
    }])
    rf_p  = int(np.round(rf.predict(row)[0]).clip(1, 3))
    xgb_p = int(np.round(xgb_model.predict(row)[0]).clip(1, 3))
    print(f"  Case {i}: {case['DayOfWeek']} | {case['Meal_Time']} | "
          f"{case['Weather']} | {case['Item']}")
    print(f"    → RF Prediction: {rf_p}  |  XGBoost Prediction: {xgb_p}\n")

print("🎉 Step 3 Complete! All charts saved to charts/ folder.")
