"""
Smart Canteen Analytics Dashboard
Modular, purple-white palette, clean typography
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Smart Canteen Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


PURPLE_PALETTE = {
    "primary":    "#6C3FC5",   
    "accent":     "#9B6FF5",   
    "light":      "#C4AAFB",  
    "lightest":   "#EDE9FF",   
    "bg":         "#F7F5FF", 
    "sidebar_bg": "#1E1040",   
    "text_dark":  "#1A0E3D",  
    "text_mid":   "#4B3A7C",  
    "text_muted": "#8A7AB5",  
    "white":      "#FFFFFF",
    "border":     "#D6CEFB",
    "chart_1":    "#6C3FC5",
    "chart_2":    "#9B6FF5",
    "chart_3":    "#C4AAFB",
    "chart_4":    "#EDE9FF",
    "warn":       "#E85D8A",
    "safe":       "#52BCA3",
    "mid":        "#F5A623",
}

CHART_COLORS = [
    "#6C3FC5","#9B6FF5","#C4AAFB","#4A2D8C",
    "#B8A0F8","#7E5CE8","#D0C4FC","#3A1F7A",
    "#A882F6","#5234A0",
]

BG_CHART = PURPLE_PALETTE["bg"]

plt.rcParams.update({
    "figure.facecolor" : BG_CHART,
    "axes.facecolor"   : BG_CHART,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.spines.left" : True,
    "axes.spines.bottom": True,
    "axes.edgecolor"   : PURPLE_PALETTE["border"],
    "font.family"      : "DejaVu Sans",
    "axes.titlesize"   : 12,
    "axes.titleweight" : "bold",
    "axes.titlecolor"  : PURPLE_PALETTE["text_dark"],
    "axes.labelsize"   : 10,
    "axes.labelcolor"  : PURPLE_PALETTE["text_mid"],
    "xtick.color"      : PURPLE_PALETTE["text_muted"],
    "ytick.color"      : PURPLE_PALETTE["text_muted"],
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
})

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
}}

/* ── Page background ── */
.stApp {{
    background-color: {PURPLE_PALETTE['bg']};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {PURPLE_PALETTE['sidebar_bg']};
    border-right: 1px solid #2D1A5E;
}}
[data-testid="stSidebar"] * {{
    color: #E8E0FF !important;
}}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {{
    color: {PURPLE_PALETTE['light']} !important;
    font-family: 'Syne', sans-serif;
}}
[data-testid="stSidebar"] hr {{
    border-color: #2D1A5E !important;
}}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background-color: {PURPLE_PALETTE['white']};
    border: 1px solid {PURPLE_PALETTE['border']};
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 12px rgba(108,63,197,0.08);
}}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {{
    color: {PURPLE_PALETTE['text_muted']} !important;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {PURPLE_PALETTE['text_dark']} !important;
    font-size: 26px;
    font-weight: 700;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background-color: {PURPLE_PALETTE['white']};
    border: 1px solid {PURPLE_PALETTE['border']};
    border-radius: 10px;
    padding: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    border-radius: 8px;
    color: {PURPLE_PALETTE['text_muted']};
    font-weight: 600;
    font-size: 14px;
    padding: 8px 22px;
    border: none;
}}
.stTabs [aria-selected="true"] {{
    background-color: {PURPLE_PALETTE['primary']} !important;
    color: {PURPLE_PALETTE['white']} !important;
    box-shadow: 0 2px 8px rgba(108,63,197,0.3);
}}

/* ── Buttons ── */
.stButton > button {{
    background-color: {PURPLE_PALETTE['primary']};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    letter-spacing: 0.02em;
    transition: all 0.2s;
    box-shadow: 0 2px 8px rgba(108,63,197,0.3);
}}
.stButton > button:hover {{
    background-color: {PURPLE_PALETTE['accent']};
    box-shadow: 0 4px 16px rgba(108,63,197,0.4);
    transform: translateY(-1px);
}}

/* ── Inputs ── */
label, .stSelectbox label, .stSlider label {{
    font-weight: 600;
    color: {PURPLE_PALETTE['text_mid']} !important;
    font-size: 13px;
    letter-spacing: 0.03em;
}}

/* ── Section headers ── */
h1 {{
    color: {PURPLE_PALETTE['text_dark']};
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 32px;
    letter-spacing: -0.02em;
}}
h2 {{
    color: {PURPLE_PALETTE['text_dark']};
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 22px;
}}
h3, h4 {{
    color: {PURPLE_PALETTE['text_mid']};
    font-weight: 600;
    font-size: 16px;
    letter-spacing: 0.01em;
}}

/* ── Divider ── */
hr {{ border-color: {PURPLE_PALETTE['border']}; }}

/* ── Alert boxes ── */
.stSuccess {{
    background-color: #DCFAF3;
    border-left: 3px solid {PURPLE_PALETTE['safe']};
    border-radius: 8px;
}}
.stInfo {{
    background-color: {PURPLE_PALETTE['lightest']};
    border-left: 3px solid {PURPLE_PALETTE['primary']};
    border-radius: 8px;
}}
.stWarning {{
    background-color: #FFF4E0;
    border-left: 3px solid {PURPLE_PALETTE['mid']};
    border-radius: 8px;
}}

/* ── Data table ── */
.stDataFrame {{
    border: 1px solid {PURPLE_PALETTE['border']};
    border-radius: 10px;
    overflow: hidden;
}}

/* ── Section card wrapper ── */
.section-card {{
    background: {PURPLE_PALETTE['white']};
    border: 1px solid {PURPLE_PALETTE['border']};
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 2px 12px rgba(108,63,197,0.06);
}}

/* ── Stat pill ── */
.stat-pill {{
    display: inline-block;
    background: {PURPLE_PALETTE['lightest']};
    color: {PURPLE_PALETTE['primary']};
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 600;
    margin: 2px;
}}
</style>
""", unsafe_allow_html=True)



@st.cache_data
def load_data():
    df = pd.read_csv("canteen_clean.csv", parse_dates=["Date"])
    df["Wastage_Risk"] = ((df["Rating"] < 3.5) & (df["Quantity"] >= 2)).astype(int)
    return df


@st.cache_resource
def train_model(df):
    le_item    = LabelEncoder()
    le_weather = LabelEncoder()
    le_meal    = LabelEncoder()
    le_day     = LabelEncoder()

    df2 = df.copy()
    df2["Item_enc"]      = le_item.fit_transform(df2["Item"])
    df2["Weather_enc"]   = le_weather.fit_transform(df2["Weather"])
    df2["Meal_Time_enc"] = le_meal.fit_transform(df2["Meal_Time"])
    df2["DayOfWeek_enc"] = le_day.fit_transform(df2["DayOfWeek"])

    FEATURES = ["DayOfWeek_enc","Meal_Time_enc","Weather_enc",
                "Hour","Item_enc","Actual_Item_Price"]
    X = df2[FEATURES]
    y = df2["Quantity"]

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf, le_item, le_weather, le_meal, le_day


df = load_data()
rf, le_item, le_weather, le_meal, le_day = train_model(df)

DAY_ORDER  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
MEAL_ORDER = ["Breakfast","Lunch","Evening Snacks"]
P = PURPLE_PALETTE



def styled_fig(w, h):
    """Return a pre-styled figure and axes."""
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG_CHART)
    ax.set_facecolor(BG_CHART)
    return fig, ax


def add_bar_labels(ax, values, fmt="{}", horizontal=False, pad=0.5, fontsize=9):
    for i, v in enumerate(values):
        text = fmt.format(int(v)) if isinstance(v, (int, float)) else fmt.format(v)
        if horizontal:
            ax.text(v + pad, i, text, va="center", fontsize=fontsize,
                    color=P["text_mid"], fontweight="600")
        else:
            ax.text(i, v + pad, text, ha="center", fontsize=fontsize,
                    color=P["text_mid"], fontweight="600")



def render_sidebar(df):
    with st.sidebar:
        st.markdown("## Smart Canteen")
        st.markdown("*Demand & Wastage Analytics*")
        st.markdown("---")

        st.markdown("### Dataset Summary")
        st.markdown(f"**Date Range**  \n{df['Date'].min().date()} to {df['Date'].max().date()}")
        st.markdown(f"**Total Records:** {len(df):,}")
        st.markdown(f"**Menu Items:** {df['Item'].nunique()}")
        st.markdown(f"**Unique Students:** {df['Student_ID'].nunique()}")
        st.markdown(f"**Avg Daily Orders:** {int(df.groupby('Date')['Quantity'].sum().mean()):,}")

        st.markdown("---")
        st.markdown("### Quick Filters")
        st.caption("Applied to the Overview tab only")
        selected_meal = st.multiselect(
            "Meal Time",
            options=MEAL_ORDER,
            default=MEAL_ORDER,
            key="sidebar_meal"
        )
        selected_weather = st.multiselect(
            "Weather",
            options=sorted(df["Weather"].unique().tolist()),
            default=sorted(df["Weather"].unique().tolist()),
            key="sidebar_weather"
        )
        st.markdown("---")
        st.markdown(
            "<span style='font-size:11px; color:#7A6BAA;'>Smart Canteen Analytics v2.0</span>",
            unsafe_allow_html=True
        )
    return selected_meal, selected_weather



def render_kpis(df):
    st.markdown("# Smart Canteen Analytics")
    st.markdown(
        "<p style='color:#8A7AB5; font-size:15px; margin-top:-10px;'>"
        "Operational intelligence for demand forecasting and food wastage reduction."
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Orders",      f"{df['Quantity'].sum():,}")
    c2.metric("Revenue Generated", f"Rs. {df['Total_Amount'].sum():,.0f}")
    c3.metric("Average Rating",    f"{df['Rating'].mean():.2f} / 5")
    c4.metric("Students Served",   f"{df['Student_ID'].nunique():,}")
    c5.metric("Wastage-Risk Rows", f"{df['Wastage_Risk'].sum():,}")

    st.markdown("---")


def render_overview(df, meal_filter, weather_filter):
    fdf = df[df["Meal_Time"].isin(meal_filter) & df["Weather"].isin(weather_filter)]
    if fdf.empty:
        st.warning("No data matches the selected filters. Adjust the sidebar filters.")
        return

    st.markdown("#### Daily Order Volume")
    daily = fdf.groupby("Date")["Quantity"].sum().reset_index()
    fig, ax = styled_fig(11, 3.2)
    ax.fill_between(daily["Date"], daily["Quantity"],
                    alpha=0.15, color=P["primary"])
    ax.plot(daily["Date"], daily["Quantity"],
            color=P["primary"], linewidth=2.2, marker="o",
            markersize=3, markerfacecolor=P["accent"])
    ax.set_xlabel("Date", labelpad=8)
    ax.set_ylabel("Quantity", labelpad=8)
    plt.xticks(rotation=25, fontsize=9)
    st.pyplot(fig, use_container_width=True)

    st.markdown("")
    col1, col2 = st.columns(2)

    # top 10 items
    with col1:
        st.markdown("#### Top 10 Items by Orders")
        top = fdf.groupby("Item")["Quantity"].sum().sort_values().tail(10)
        fig, ax = styled_fig(6, 4.5)
        bars = ax.barh(top.index, top.values,
                       color=CHART_COLORS[:10], edgecolor="white", linewidth=0.8,
                       height=0.65)
        add_bar_labels(ax, top.values, horizontal=True, pad=5)
        ax.set_xlabel("Total Quantity")
        ax.set_xlim(0, top.values.max() * 1.15)
        st.pyplot(fig, use_container_width=True)

    
    with col2:
        st.markdown("#### Orders by Day of Week")
        day_d = fdf.groupby("DayOfWeek")["Quantity"].sum().reindex(DAY_ORDER).dropna()
        fig, ax = styled_fig(6, 4.5)
        bars = ax.bar(day_d.index, day_d.values,
                      color=P["accent"], edgecolor="white", linewidth=0.8,
                      width=0.55)
        add_bar_labels(ax, day_d.values, pad=3, fontsize=9)
        plt.xticks(rotation=20, fontsize=9)
        ax.set_ylabel("Quantity")
        ax.set_ylim(0, day_d.max() * 1.15)
        st.pyplot(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Peak Order Hours")
        hourly = fdf.groupby("Hour")["Quantity"].sum()
        peak_hours = [8, 9, 12, 13, 16, 17]
        bar_colors = [P["primary"] if h in peak_hours else P["light"]
                      for h in hourly.index]
        fig, ax = styled_fig(6, 3.5)
        ax.bar(hourly.index, hourly.values,
               color=bar_colors, edgecolor="white", linewidth=0.5, width=0.7)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Quantity")
        peak_patch  = mpatches.Patch(color=P["primary"], label="Peak hours")
        other_patch = mpatches.Patch(color=P["light"],   label="Off-peak")
        ax.legend(handles=[peak_patch, other_patch], fontsize=9, framealpha=0)
        st.pyplot(fig, use_container_width=True)


    with col4:
        st.markdown("#### Payment Mode Distribution")
        pay = fdf["Payment_Mode"].value_counts()
        fig, ax = styled_fig(5, 3.5)
        wedges, texts, autotexts = ax.pie(
            pay.values,
            labels=pay.index,
            autopct="%1.1f%%",
            colors=[P["primary"], P["accent"], P["light"]],
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            textprops=dict(fontsize=10),
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_color(P["text_dark"])
        st.pyplot(fig, use_container_width=True)

  
    st.markdown("#### Average Demand by Weather Condition")
    weather_d = fdf.groupby("Weather")["Quantity"].mean().sort_values(ascending=False)
    fig, ax = styled_fig(10, 3)
    colors = [P["primary"] if i == 0 else P["accent"] if i == 1 else P["light"]
              for i in range(len(weather_d))]
    ax.bar(weather_d.index, weather_d.values,
           color=colors, edgecolor="white", linewidth=0.8, width=0.5)
    for i, v in enumerate(weather_d.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9.5,
                color=P["text_mid"], fontweight="600")
    ax.set_ylabel("Avg Quantity per Transaction")
    ax.set_ylim(0, weather_d.max() * 1.2)
    st.pyplot(fig, use_container_width=True)



def render_prediction(df):
    st.markdown("#### Configure Prediction Parameters")
    st.markdown(
        "<p style='color:#8A7AB5; font-size:13px;'>"
        "Select the operating conditions and receive a demand forecast from the trained Random Forest model."
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    with c1:
        sel_day  = st.selectbox("Day of Week", DAY_ORDER)
        sel_meal = st.selectbox("Meal Time",   MEAL_ORDER)
    with c2:
        sel_weather = st.selectbox("Weather Condition", df["Weather"].unique().tolist())
        sel_item    = st.selectbox("Menu Item", sorted(df["Item"].unique().tolist()))
    with c3:
        sel_hour  = st.slider("Hour of Day",   8, 20, 12)
        sel_price = st.slider(
            "Item Price (Rs.)",
            int(df["Actual_Item_Price"].min()),
            int(df["Actual_Item_Price"].max()),
            int(df["Actual_Item_Price"].median())
        )

    st.markdown("")
    if st.button("Run Demand Forecast"):
        row = pd.DataFrame([{
            "DayOfWeek_enc"    : le_day.transform([sel_day])[0],
            "Meal_Time_enc"    : le_meal.transform([sel_meal])[0],
            "Weather_enc"      : le_weather.transform([sel_weather])[0],
            "Hour"             : sel_hour,
            "Item_enc"         : le_item.transform([sel_item])[0],
            "Actual_Item_Price": sel_price,
        }])
        pred  = int(np.round(rf.predict(row)[0]).clip(1, 3))
        label = {1: "Low", 2: "Medium", 3: "High"}[pred]
        level = {1: P["safe"], 2: P["mid"], 3: P["warn"]}[pred]

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Quantity", f"{pred} unit(s)")
        r2.metric("Demand Level",       label)
        r3.metric("Estimated Revenue",  f"Rs. {pred * sel_price}")

        msg = {
            1: ("Demand is expected to be low. Prepare a smaller batch to minimise waste.", "success"),
            2: ("Moderate demand is expected. A standard batch size is recommended.", "info"),
            3: ("High demand is expected. Prepare additional stock to avoid shortages.", "warning"),
        }
        text, kind = msg[pred]
        if kind == "success":
            st.success(text)
        elif kind == "info":
            st.info(text)
        else:
            st.warning(text)

    st.markdown("---")
    st.markdown("#### Demand Forecast Across All Menu Items")
    st.caption("Baseline scenario — Monday, Lunch, Sunny, 12:00")

    rows = []
    for item in sorted(df["Item"].unique().tolist()):
        price = int(df[df["Item"] == item]["Actual_Item_Price"].mode()[0])
        r = pd.DataFrame([{
            "DayOfWeek_enc"    : le_day.transform(["Monday"])[0],
            "Meal_Time_enc"    : le_meal.transform(["Lunch"])[0],
            "Weather_enc"      : le_weather.transform(["Sunny"])[0],
            "Hour"             : 12,
            "Item_enc"         : le_item.transform([item])[0],
            "Actual_Item_Price": price,
        }])
        p = int(np.round(rf.predict(r)[0]).clip(1, 3))
        rows.append({
            "Item"          : item,
            "Price (Rs.)"   : price,
            "Predicted Qty" : p,
            "Demand Level"  : ["Low","Medium","High"][p - 1],
        })

    tdf = pd.DataFrame(rows)
    st.dataframe(tdf, use_container_width=True, hide_index=True)



def render_wastage(df):
    total_risk = df["Wastage_Risk"].sum()
    risk_pct   = total_risk / len(df) * 100

    w1, w2, w3 = st.columns(3)
    w1.metric("High-Risk Transactions", f"{total_risk:,}")
    w2.metric("Overall Wastage Risk",   f"{risk_pct:.1f}%")

    # Safest item from data
    iw_all = (df.groupby("Item")["Wastage_Risk"].agg(["sum","count"]))
    iw_all["pct"] = iw_all["sum"] / iw_all["count"] * 100
    safest = iw_all["pct"].idxmin()
    w3.metric("Lowest-Risk Item", safest)

    st.markdown("")
    col1, col2 = st.columns(2)

    # Wastage by item
    with col1:
        st.markdown("#### Wastage Risk by Menu Item")
        iw = iw_all.sort_values("pct", ascending=True)
        bar_c = [P["warn"] if v >= 4 else P["mid"] if v >= 2 else P["safe"]
                 for v in iw["pct"]]
        fig, ax = styled_fig(6, 6)
        ax.barh(iw.index, iw["pct"], color=bar_c, edgecolor="white",
                linewidth=0.8, height=0.65)
        ax.axvline(iw["pct"].mean(), color=P["primary"],
                   linestyle="--", linewidth=1.5,
                   label=f"Average: {iw['pct'].mean():.1f}%")
        patches = [
            mpatches.Patch(color=P["warn"], label="High  >= 4%"),
            mpatches.Patch(color=P["mid"],  label="Medium 2-4%"),
            mpatches.Patch(color=P["safe"], label="Low    < 2%"),
        ]
        ax.legend(handles=patches, fontsize=9, loc="lower right", framealpha=0)
        ax.set_xlabel("Wastage Risk %")
        st.pyplot(fig, use_container_width=True)

    with col2:
        # Wastage by item
        st.markdown("#### Wastage Risk by Day of Week")
        dw = (df.groupby("DayOfWeek")["Wastage_Risk"]
                .agg(["sum","count"]))
        dw["pct"] = dw["sum"] / dw["count"] * 100
        dw = dw.reindex(DAY_ORDER).dropna()
        bar_c = [P["warn"] if v >= 4 else P["mid"] if v >= 2 else P["safe"]
                 for v in dw["pct"]]
        fig, ax = styled_fig(6, 3.2)
        ax.bar(dw.index, dw["pct"], color=bar_c,
               edgecolor="white", linewidth=0.8, width=0.55)
        for i, v in enumerate(dw["pct"]):
            ax.text(i, v + 0.08, f"{v:.1f}%",
                    ha="center", fontsize=9, fontweight="600",
                    color=P["text_mid"])
        ax.axhline(dw["pct"].mean(), color=P["primary"],
                   linestyle="--", linewidth=1.5, alpha=0.7)
        plt.xticks(rotation=20, fontsize=9)
        ax.set_ylabel("Wastage Risk %")
        st.pyplot(fig, use_container_width=True)

        st.markdown("")

        #Wastage by Meal Time
        st.markdown("#### Wastage Risk by Meal Time")
        mw = (df.groupby("Meal_Time")["Wastage_Risk"]
                .agg(["sum","count"]))
        mw["pct"] = mw["sum"] / mw["count"] * 100
        bar_c = [P["warn"] if v >= 4 else P["mid"] if v >= 2 else P["safe"]
                 for v in mw["pct"]]
        fig, ax = styled_fig(6, 2.8)
        ax.bar(mw.index, mw["pct"], color=bar_c,
               edgecolor="white", linewidth=0.8, width=0.45)
        for i, v in enumerate(mw["pct"]):
            ax.text(i, v + 0.08, f"{v:.1f}%",
                    ha="center", fontsize=10, fontweight="600",
                    color=P["text_mid"])
        ax.set_ylabel("Wastage Risk %")
        st.pyplot(fig, use_container_width=True)

    # Full Table
    st.markdown("#### Complete Wastage Risk Breakdown")
    iw_table = iw_all[["sum","count","pct"]].copy()
    iw_table.columns = ["Risk Transactions","Total Transactions","Risk %"]
    iw_table["Risk %"]      = iw_table["Risk %"].round(2)
    iw_table["Avg Rating"]  = df.groupby("Item")["Rating"].mean().round(2)
    iw_table["Risk Level"]  = iw_table["Risk %"].apply(
        lambda x: "High" if x >= 4 else ("Medium" if x >= 2 else "Low")
    )
    iw_table = iw_table.sort_values("Risk %", ascending=False)
    st.dataframe(iw_table, use_container_width=True)

    # Recommendations
    st.markdown("---")
    st.markdown("#### Operational Recommendations")

    top2_risky = iw_all.sort_values("pct", ascending=False).head(2).index.tolist()
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.info(
            f"**Reduce batch size** for *{top2_risky[0]}* and *{top2_risky[1]}* — "
            f"these items carry the highest wastage risk across the dataset."
        )
        st.info(
            "**Wednesday lunch** is the peak wastage window. "
            "Consider scaling down preparation quantities for that slot."
        )
    with rec_col2:
        st.success(
            f"**{safest}** has the lowest wastage risk and consistent demand — "
            "a reliable staple to maintain in standard quantities."
        )
        st.success(
            "**Sunny-day orders** are highest on average. "
            "Use weather forecasts to pre-plan production quantities the night before."
        )



def main():
    selected_meal, selected_weather = render_sidebar(df)
    render_kpis(df)

    tab1, tab2, tab3 = st.tabs(["Overview", "Demand Prediction", "Wastage Analysis"])

    with tab1:
        render_overview(df, selected_meal, selected_weather)

    with tab2:
        render_prediction(df)

    with tab3:
        render_wastage(df)


if __name__ == "__main__":
    main()