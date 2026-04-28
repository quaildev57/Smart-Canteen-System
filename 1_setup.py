import pandas as pd
import numpy as np

df = pd.read_csv("college_canteen_dataset_v8.csv")

#dates & times
df["Date"]      = pd.to_datetime(df["Date"], dayfirst=True)
df["Time"]      = pd.to_datetime(df["Time"], format="%H:%M").dt.time
df["Hour"]      = pd.to_datetime(df["Time"].astype(str)).dt.hour
df["Month"]     = df["Date"].dt.month
df["Week"]      = df["Date"].dt.isocalendar().week.astype(int)

# encoding
df["DayOfWeek_enc"]  = pd.Categorical(df["DayOfWeek"],
                        categories=["Monday","Tuesday","Wednesday",
                                    "Thursday","Friday","Saturday"],
                        ordered=True).codes
df["Meal_Time_enc"]  = pd.Categorical(df["Meal_Time"],
                        categories=["Breakfast","Lunch","Evening Snacks"],
                        ordered=True).codes
df["Weather_enc"]    = pd.factorize(df["Weather"])[0]
df["Item_enc"]       = pd.factorize(df["Item"])[0]

# Low rating (<3.5) + high quantity = likely leftover / dissatisfied
df["Wastage_Risk"] = ((df["Rating"] < 3.5) & (df["Quantity"] >= 2)).astype(int)

print("=" * 55)
print("  SMART CANTEEN — DATASET SUMMARY")
print("=" * 55)
print(f"  Total Transactions : {len(df):,}")
print(f"  Date Range         : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"  Unique Students    : {df['Student_ID'].nunique()}")
print(f"  Unique Items       : {df['Item'].nunique()}")
print(f"  Meal Types         : {df['Meal_Time'].unique().tolist()}")
print(f"  Weather Types      : {df['Weather'].unique().tolist()}")
print(f"  Payment Modes      : {df['Payment_Mode'].unique().tolist()}")
print(f"  Avg Rating         : {df['Rating'].mean():.2f} / 5.0")
print(f"  Avg Order Value    : ₹{df['Total_Amount'].mean():.1f}")
print(f"  Total Revenue      : ₹{df['Total_Amount'].sum():,}")
print(f"  High Wastage Risk  : {df['Wastage_Risk'].sum()} transactions "
      f"({df['Wastage_Risk'].mean()*100:.1f}%)")
print("=" * 55)
print("\nColumn Overview:")
print(df.dtypes)
print("\nFirst 3 rows:")
print(df[["Date","Meal_Time","Item","Quantity","Weather",
          "Rating","Total_Amount","Wastage_Risk"]].head(3).to_string(index=False))

df.to_csv("canteen_clean.csv", index=False)
print("\n✅ Cleaned dataset saved → canteen_clean.csv")
