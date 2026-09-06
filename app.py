# app.py
"""
Streamlit app that uses visualizations.py to display an interactive dashboard.

Usage:
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import visualizations as vz
import os
from typing import Optional

st.set_page_config(page_title="Credit Card Fraud Dashboard", layout="wide", initial_sidebar_state="expanded")

# --------------------
# Data loading & cache
# --------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # gracefully drop unnamed index columns if present
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df

@st.cache_data
def load_and_preprocess(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        default_path = "./credit-card-transactions.csv"
        if not os.path.exists(default_path):
            # return empty df if default missing
            return pd.DataFrame()
        df = load_csv(default_path)
    # Basic cleaning steps that mirror your original notebook
    # convert timestamp column if present
    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        df["year"] = df["trans_date_trans_time"].dt.year
        df["month"] = df["trans_date_trans_time"].dt.month
        df["day"] = df["trans_date_trans_time"].dt.day  # numeric day of month
        df["weekday"] = df["trans_date_trans_time"].dt.day_name()  # Monday, Tuesday, etc.
        df["hour"] = df["trans_date_trans_time"].dt.hour

    # Try to coerce amt numeric
    if "amt" in df.columns:
        df["amt"] = pd.to_numeric(df["amt"], errors="coerce")
    # extract year/month/hour etc via viz.preprocess_df
    df = vz.preprocess_df(df)
    return df

# --------------------
# Sidebar - Inputs
# --------------------
st.sidebar.title("Data & Filters")

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
df = load_and_preprocess(uploaded)

if df.empty:
    st.sidebar.warning("No data loaded. Place 'credit_card_transactions.csv' in the app folder or upload a CSV.")
    st.write("No data available. Upload a CSV or place 'credit_card_transactions.csv' in this directory.")
    st.stop()

# derived lists
all_cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
all_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Sidebar controls
with st.sidebar.expander("Quick filters", expanded=True):
    # Date range
    if "trans_date_trans_time" in df.columns:
        min_date = df["trans_date_trans_time"].min().date()
        max_date = df["trans_date_trans_time"].max().date()
        date_range = st.date_input("Transaction date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        date_range = None

    # Fraud filter
    if "is_fraud" in df.columns:
        fraud_filter = st.multiselect("Show is_fraud values", options=sorted(df["is_fraud"].dropna().unique().tolist()), default=sorted(df["is_fraud"].dropna().unique().tolist()))
    else:
        fraud_filter = None

    # Top N
    top_n = st.sidebar.slider("Top N categories", min_value=3, max_value=30, value=10)

    # Categorical columns to visualize
    select_cat_cols = st.sidebar.multiselect("Categorical columns to analyze (left as blank to auto-select top 6)", options=all_cat_cols, default=None)

    # Numeric columns for correlation
    select_num_cols = st.sidebar.multiselect("Numeric columns for correlation", options=all_num_cols, default=None)

    # Job minimum transactions
    min_job_txn = st.sidebar.number_input("Min transactions (jobs chart)", min_value=1, max_value=1000, value=50, step=1)

# Apply filters to df (create a working copy)
df_work = df.copy()
# Date filter
if date_range and len(date_range) == 2:
    start_d, end_d = date_range
    df_work = df_work[(df_work["trans_date_trans_time"].dt.date >= start_d) & (df_work["trans_date_trans_time"].dt.date <= end_d)]

# Fraud filter
if fraud_filter is not None and "is_fraud" in df_work.columns:
    df_work = df_work[df_work["is_fraud"].isin(fraud_filter)]

# If user has not selected categorical columns, auto-pick up to 6 useful ones
if not select_cat_cols:
    guessed = [c for c in ["category", "job", "merchant", "gender", "city", "state"] if c in all_cat_cols]
    select_cat_cols = guessed[:6] if guessed else all_cat_cols[:6]

# numeric columns for correlations
if not select_num_cols:
    # pick typical numeric columns if exist
    fallback_num = [c for c in ["amt", "credit_limit", "balance", "age", "num_trans"] if c in all_num_cols]
    select_num_cols = fallback_num if fallback_num else (all_num_cols[:6] if all_num_cols else [])

# --------------------
# Top KPIs
# --------------------
st.title("Credit Card Fraud Insights")
k1, k2, k3, k4 = st.columns(4)
total_txns = len(df_work)
total_frauds = int(df_work["is_fraud"].sum()) if "is_fraud" in df_work.columns else np.nan
fraud_rate = (total_frauds / total_txns * 100) if total_txns else 0
total_amount = df_work["amt"].sum() if "amt" in df_work.columns else np.nan
avg_amount = df_work["amt"].mean() if "amt" in df_work.columns else np.nan

k1.metric("Total Transactions", f"{total_txns:,}")
k2.metric("Total Frauds", f"{total_frauds:,}", delta=f"{fraud_rate:.2f}%")
k3.metric("Total Amount", f"{total_amount:,.2f}" if pd.notna(total_amount) else "N/A")
k4.metric("Avg Amount", f"{avg_amount:,.2f}" if pd.notna(avg_amount) else "N/A")

# --------------------
# Tabs (layout of insights)
# --------------------
tab_overview, tab_demographics, tab_categories, tab_time, tab_jobs, tab_corr = st.tabs(
    ["Overview", "Demographics & Age", "Categories", "Time Analysis", "Jobs & Occupation", "Correlations & Summary"]
)

# ===============
# OVERVIEW TAB
# ===============
with tab_overview:
    st.header("Overview")
    
    st.subheader("Monthly Transaction Trends")
    fig_month = vz.plot_monthly_trends(df_work)
    st.plotly_chart(fig_month, use_container_width=True)
    st.subheader("Fraud Rate Over Time (%)")
    fig_fraud_time = vz.plot_fraud_rate_over_time(df_work)
    st.plotly_chart(fig_fraud_time, use_container_width=True)
    
    st.subheader("Hourly Average Spend")
    st.plotly_chart(vz.plot_avg_spend_by_hour(df_work), use_container_width=True)
    st.subheader("Hourly Trends: Normal vs Fraud")
    st.plotly_chart(vz.plot_hourly_trends_normal_and_fraud(df_work), use_container_width=True)

# =========================
# DEMOGRAPHICS & AGE TAB
# =========================
with tab_demographics:
    st.header("Demographics & Age-based Insights")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Age-based transaction count & average")
        st.plotly_chart(vz.plot_age_transaction_dual(df_work), use_container_width=True)
        st.subheader("Average Transaction Amount per Age Bin")
        st.plotly_chart(vz.plot_avg_amount_by_age_bin(df_work), use_container_width=True)
    with c2:
        st.subheader("Fraud Rate per Age Bin")
        st.plotly_chart(vz.plot_age_bin_fraud_rate_area(df_work), use_container_width=True)
        st.subheader("Average Fraud Amount per Age Bin")
        st.plotly_chart(vz.plot_avg_fraud_amount_by_age_bin(df_work), use_container_width=True)

    st.markdown("### Transaction Category distribution per age bin")
    st.plotly_chart(vz.plot_category_age_heatmap(df_work), use_container_width=True)

# =========================
# CATEGORIES TAB
# =========================
with tab_categories:
    st.header("Category Analysis")

    # Show top categories for each selected cat col in two columns layout
    st.subheader("Top Categories")
    cols = st.columns(2)
    for idx, col_name in enumerate(select_cat_cols):
        fig = vz.plot_top_categories(df_work, col_name, top_n=top_n, color="Greens")
        cols[idx % 2].plotly_chart(fig, use_container_width=True)

    st.subheader("Fraud Rate by Category")
    # show fraud rate figs in a grid
    fraud_figs = vz.plot_fraud_rate_list(df_work, select_cat_cols, target="is_fraud", top_n=top_n, color="Greens")
    # display them in 2-column grid
    if fraud_figs:
        i = 0
        cols = st.columns(2)
        for col_name, fig in fraud_figs.items():
            cols[i % 2].plotly_chart(fig, use_container_width=True)
            i += 1

# =========================
# TIME ANALYSIS TAB
# =========================
with tab_time:
    st.header("Time-based analysis")

    st.subheader("Fraudulent transactions by weekday")
    st.plotly_chart(vz.plot_weekday_fraud_bar(df_work, color="Purples"), use_container_width=True)

    st.subheader("Fraud rate heatmap by weekday & hour")
    st.plotly_chart(vz.plot_fraud_rate_heatmap_weekday_hour(df_work, color="Purples"), use_container_width=True)

    st.subheader("Hourly trends (normal vs fraud)")
    st.plotly_chart(vz.plot_hourly_trends_normal_and_fraud(df_work, color="Purples"), use_container_width=True)


# =========================
# JOBS & OCCUPATION TAB
# =========================
with tab_jobs:
    st.header("Jobs & Occupation Insights")
    st.subheader("Top Jobs by Fraud Rate")
    st.plotly_chart(vz.plot_top_jobs_by_fraud_rate(df_work, job_col="job", target="is_fraud", min_txn=min_job_txn, color='Purples'), use_container_width=True)

    st.subheader("Top Job Categories per Age Bin (counts heatmap)")
    st.plotly_chart(vz.plot_job_age_heatmap(df_work, job_col="job", color='Viridis'), use_container_width=True)

    st.subheader("Top Job per Age Bin (by average amount)")
    st.plotly_chart(vz.plot_top_job_per_age_bin_avg_amount(df_work, color='Purples'), use_container_width=True)

    st.subheader("Top Job per Age Bin (by transaction count)")
    st.plotly_chart(vz.plot_top_job_per_age_bin_count(df_work, color='Purples'), use_container_width=True)

# =========================
# CORRELATIONS & SUMMARY
# =========================
with tab_corr:
    st.header("Correlations & Summary Stats")
    st.subheader("Correlation heatmap (numeric columns)")
    st.plotly_chart(vz.plot_correlation_heatmap(df_work, select_num_cols), use_container_width=True)

    st.subheader("Summary statistics by target")
    st.plotly_chart(vz.plot_summary_stats_by_target_heatmap(df_work, target="is_fraud", cols=select_num_cols), use_container_width=True)

# -------------------------
# Footer / quick tips
# -------------------------
st.markdown("---")
st.caption("Tip: upload a CSV in the left sidebar to run the dashboard on any dataset with similar column names. If your column names differ, adjust the sidebar selections accordingly.")
