# visualizations.py
"""
Plotly visualizations for credit card transaction fraud analysis.

All plotting functions accept a `color` parameter (default "Blues") so you can
swap themes easily (e.g., "Purples", "Greens", "Viridis", "Inferno", etc.).

Each function returns a Plotly Figure which the Streamlit app can render with
`st.plotly_chart(..., use_container_width=True)`.

Assumes the dataframe has been preprocessed with `preprocess_df()` (or you can
call it before passing the df to the plotting functions).
"""
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pcolors


# -------------------------
# Utilities / Preprocessing
# -------------------------
def preprocess_df(df: pd.DataFrame, date_col: str = "trans_date_trans_time") -> pd.DataFrame:
    """
    Add commonly-used columns to the dataframe and return a copy.
    - ensures date_col is datetime
    - adds year, month, day, hour, weekday, weekday_name
    - computes age from dob (if present) and creates age bins/ordering
    """
    df = df.copy()

    # Ensure date column
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df[date_col] = pd.NaT

    # Time features
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["hour"] = df[date_col].dt.hour
    df["weekday"] = df[date_col].dt.weekday  # 0 = Monday
    df["weekday_name"] = df[date_col].dt.day_name()

    # Age (from dob)
    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
        ref_dates = df[date_col].fillna(pd.Timestamp.today())
        df["age"] = ((ref_dates - df["dob"]).dt.days / 365.25).apply(lambda x: int(x) if pd.notna(x) else np.nan)
    else:
        df["age"] = np.nan

    # Age bins (10-90, step 10) and ordering
    try:
        bins = list(range(10, 100, 10))
        if df["age"].notna().any():
            df["age_bin"] = pd.cut(df["age"].replace({np.nan: None}), bins=bins, right=False)
            df["age_bin_left"] = df["age_bin"].apply(lambda x: x.left if pd.notna(x) else np.nan)
            df["age_bin_str"] = df["age_bin"].astype(str)
            valid_bins = df.dropna(subset=["age_bin_str", "age_bin_left"])

            age_bin_order = (
                valid_bins[["age_bin_str", "age_bin_left"]]
                .drop_duplicates()
                .sort_values("age_bin_left")["age_bin_str"]
                .tolist()
            )

            df.attrs["age_bin_order"] = age_bin_order
        else:
            df["age_bin"] = pd.NA
            df["age_bin_left"] = np.nan
            df["age_bin_str"] = pd.NA
            df.attrs["age_bin_order"] = []
    except Exception:
        df["age_bin"] = pd.NA
        df["age_bin_left"] = np.nan
        df["age_bin_str"] = pd.NA
        df.attrs["age_bin_order"] = []

    return df


def _empty_fig(message: str = "No data to display") -> go.Figure:
    """Return a simple Plotly figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=300, margin=dict(t=20, b=20))
    return fig


def _get_palette(scale_name: str) -> Optional[List[str]]:
    """
    Try to resolve a named Plotly palette into a list of colors.
    Returns None if not found.
    """
    if not scale_name:
        return None

    # Try sequential, diverging, cyclical, qualitative groups
    groups = ("sequential", "diverging", "cyclical", "qualitative")
    for g in groups:
        grp = getattr(pcolors, g, None)
        if grp and hasattr(grp, scale_name):
            pal = getattr(grp, scale_name)
            if isinstance(pal, (list, tuple)) and pal:
                return list(pal)
    return None


def _sample_color_from_scale(scale_name: str, prefer_index: Optional[int] = None) -> str:
    """
    Return a single color string sampled from a named scale.
    If the scale isn't found, return `scale_name` (useful if it's a CSS color/hex).
    """
    pal = _get_palette(scale_name)
    if pal:
        idx = prefer_index if prefer_index is not None else len(pal) // 2
        idx = max(0, min(idx, len(pal) - 1))
        return pal[idx]
    # fallback: perhaps the user passed a hex or color name
    return scale_name


# -------------------------
# Plotting functions
# -------------------------
DEFAULT_COLOR = "Blues"


def plot_top_categories(df: pd.DataFrame, col: str, top_n: int = 15, color: str = DEFAULT_COLOR) -> go.Figure:
    """
    Bar chart for top categories in `col`. Uses the counts as a continuous color scale.
    """
    if col not in df.columns:
        return _empty_fig(f"Column '{col}' not found")

    counts = df[col].value_counts().nlargest(top_n).reset_index()
    counts.columns = [col, "count"]

    if counts.empty:
        return _empty_fig(f"No categories in '{col}'")

    fig = px.bar(
        counts,
        x=col,
        y="count",
        text="count",
        title=f"Top {top_n} categories in {col.capitalize()}",
        color="count",
        color_continuous_scale=color,
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-45, height=420, showlegend=False, margin=dict(t=40, b=80))
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str], color: str = DEFAULT_COLOR) -> go.Figure:
    """
    Correlation heatmap for numeric_cols. Uses the provided color scale.
    """
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 2:
        return _empty_fig("Need at least two numeric columns for correlation heatmap")
    corr = df[cols].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Heatmap of Numeric Variables", zmin=-1, zmax=1, color_continuous_scale=color)
    fig.update_layout(height=600, margin=dict(t=40, b=40))
    return fig


def plot_fraud_rate_by_category(df: pd.DataFrame, col: str, target: str = "is_fraud", top_n: int = 10, color: str = DEFAULT_COLOR) -> go.Figure:
    """
    Bar chart showing fraud rate (mean of `target`) for top categories in `col`.
    """
    if col not in df.columns:
        return _empty_fig(f"Column '{col}' not found")
    if target not in df.columns:
        return _empty_fig(f"Target '{target}' not found")

    top_categories = df[col].value_counts().nlargest(top_n).index.tolist()
    filtered = df[df[col].isin(top_categories)]
    if filtered.empty:
        return _empty_fig("No data for selected categories")

    fraud_rate = filtered.groupby(col)[target].mean().reset_index().sort_values(target, ascending=False)
    fig = px.bar(
        fraud_rate,
        x=col,
        y=target,
        text=target,
        title=f"{col}: Fraud Rate (Top {top_n})",
        color=target,
        color_continuous_scale=color,
    )

    # display as percentage
    fig.update_traces(texttemplate="%{y:.2%}", textposition="outside")
    fig.update_layout(yaxis_tickformat=".1%", xaxis_tickangle=-45, height=420, showlegend=False, margin=dict(t=40, b=80))
    return fig


def plot_fraud_rate_list(df: pd.DataFrame, cat_cols: List[str], target: str = "is_fraud", top_n: int = 10, color: str = DEFAULT_COLOR) -> Dict[str, go.Figure]:
    """
    Return a mapping of column -> fraud-rate figure for quick grid rendering.
    """
    figs: Dict[str, go.Figure] = {}
    for col in cat_cols:
        if col == target:
            continue
        figs[col] = plot_fraud_rate_by_category(df, col, target=target, top_n=top_n, color=color)
    return figs


def plot_summary_stats_by_target_heatmap(df: pd.DataFrame, target: str = "is_fraud", cols: Optional[List[str]] = None, color: str = DEFAULT_COLOR) -> go.Figure:
    """
    Heatmap of summary stats (mean, median, std) for numeric columns split by target.
    """
    if target not in df.columns:
        return _empty_fig(f"Target '{target}' not found")

    if cols is None:
        cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != target]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return _empty_fig("No numeric columns found")

    stats = df.groupby(target)[cols].agg(["mean", "median", "std"])
    stats.columns = ["_".join(col).strip() for col in stats.columns.values]
    matrix = stats.T  # rows = metric_col, columns = target values

    fig = px.imshow(matrix, text_auto=".1f", aspect="auto", title=f"Summary Statistics of Numeric Columns by {target}", color_continuous_scale=color)
    fig.update_layout(height=500, margin=dict(t=40, b=40))
    return fig


def plot_top_jobs_by_fraud_rate(df: pd.DataFrame, job_col: str = "job", target: str = "is_fraud", min_txn: int = 50, top_n: int = 10, color: str = DEFAULT_COLOR) -> go.Figure:
    """
    Top jobs by fraud rate (only consider jobs with at least min_txn transactions).
    """
    if job_col not in df.columns:
        return _empty_fig(f"Column '{job_col}' not found")
    if target not in df.columns:
        return _empty_fig(f"Target '{target}' not found")

    fraud_stats_by_job = (
        df.groupby(job_col)[target]
        .agg(["mean", "count", "sum"])
        .rename(columns={"mean": "fraud_rate", "count": "total_txns", "sum": "fraud_txns"})
        .reset_index()
    )

    fraud_stats_by_job = fraud_stats_by_job[fraud_stats_by_job["total_txns"] >= min_txn]
    if fraud_stats_by_job.empty:
        return _empty_fig("No jobs meet the minimum transaction threshold")

    fraud_stats_by_job["fraud_rate_pct"] = fraud_stats_by_job["fraud_rate"] * 100
    top_jobs = fraud_stats_by_job.sort_values("fraud_rate_pct", ascending=False).head(top_n)

    fig = px.bar(
        top_jobs,
        x=job_col,
        y="fraud_rate_pct",
        text="fraud_rate_pct",
        title=f"Top {top_n} Jobs by Fraud Rate (Min {min_txn} Txns)",
        color="fraud_rate_pct",
        color_continuous_scale=color,
    )

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(xaxis_tickangle=-45, height=480, showlegend=False, margin=dict(t=40, b=80))
    return fig


def plot_age_transaction_dual(df: pd.DataFrame, color: str = DEFAULT_COLOR) -> go.Figure:
    """
    Dual-axis chart: bar = transaction count, line = average amount per age bin.
    """
    if "age_bin_str" not in df.columns or "amt" not in df.columns:
        return _empty_fig("Needed columns (age_bin_str, amt) not found")
    df2 = df.dropna(subset=["age_bin_str"])
    if df2.empty:
        return _empty_fig("No age-bin data")

    agg = df2.groupby("age_bin_str").agg(transaction_count=("amt", "count"), average_amount=("amt", "mean")).reset_index()
    age_bin_order = df.attrs.get("age_bin_order", agg["age_bin_str"].tolist())
    agg["age_bin_str"] = pd.Categorical(agg["age_bin_str"], categories=age_bin_order, ordered=True)
    agg = agg.sort_values("age_bin_str")

    palette = _get_palette(color)
    bar_color = _sample_color_from_scale(color, prefer_index=len(palette)//2 if palette else None)
    line_color = _sample_color_from_scale(color, prefer_index=0 if palette else None)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=agg["age_bin_str"].astype(str), y=agg["transaction_count"], name="Transaction Count", marker_color=bar_color), secondary_y=False)
    fig.add_trace(go.Scatter(x=agg["age_bin_str"].astype(str), y=agg["average_amount"], mode="lines+markers", name="Average Amount", line=dict(color=line_color)), secondary_y=True)

    fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
    fig.update_yaxes(title_text="Average Transaction Amount", secondary_y=True)
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(title="Age-Based Transaction Analysis", height=480, margin=dict(t=40, b=80))
    return fig


def plot_age_bin_fraud_rate_area(df: pd.DataFrame, target: str = "is_fraud", color: str = DEFAULT_COLOR) -> go.Figure:
    """
    Area chart showing fraud rate (%) per age bin.
    """
    if "age_bin_str" not in df.columns or target not in df.columns:
        return _empty_fig("Required columns not found")
    df2 = df.dropna(subset=["age_bin_str"])
    if df2.empty:
        return _empty_fig("No age data")

    fraud_rate = df2.groupby("age_bin_str").agg(fraud_rate=(target, "mean")).reset_index()
    age_bin_order = df.attrs.get("age_bin_order", fraud_rate["age_bin_str"].tolist())
    fraud_rate["age_bin_str"] = pd.Categorical(fraud_rate["age_bin_str"], categories=age_bin_order, ordered=True)
    fraud_rate = fraud_rate.sort_values("age_bin_str")
    fraud_rate["fraud_pct"] = fraud_rate["fraud_rate"] * 100

    fig = px.area(fraud_rate, x="age_bin_str", y="fraud_pct", markers=True, title="Fraud Rate per Age Bin (%)", color_discrete_sequence=[_sample_color_from_scale(color)])
    fig.update_layout(yaxis_title="Fraud Rate (%)", xaxis_title="Age Bin", height=420, margin=dict(t=40, b=80))
    return fig


def plot_avg_amount_by_age_bin(df: pd.DataFrame, color: str = DEFAULT_COLOR) -> go.Figure:
    """
    Bar chart: average transaction amount per age bin.
    """
    if "age_bin_str" not in df.columns or "amt" not in df.columns:
        return _empty_fig("Required columns not found")
    df2 = df.dropna(subset=["age_bin_str"])
    agg = df2.groupby("age_bin_str").agg(avg_amount=("amt", "mean")).reset_index()
    age_bin_order = df.attrs.get("age_bin_order", agg["age_bin_str"].tolist())
    agg["age_bin_str"] = pd.Categorical(agg["age_bin_str"], categories=age_bin_order, ordered=True)
    agg = agg.sort_values("age_bin_str")

    fig = px.bar(agg, x="age_bin_str", y="avg_amount", title="Average Transaction Amount per Age Bin", text="avg_amount", color_discrete_sequence=[_sample_color_from_scale(color)])
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(xaxis_tickangle=-45, height=420, margin=dict(t=40, b=80))
    return fig


def plot_avg_fraud_amount_by_age_bin(df: pd.DataFrame, target: str = "is_fraud", color: str = DEFAULT_COLOR) -> go.Figure:
    """
    Bar chart: average fraud amount per age bin (fraud cases only).
    """
    if "age_bin_str" not in df.columns or "amt" not in df.columns or target not in df.columns:
        return _empty_fig("Required columns not found")
    df2 = df[df[target] == 1].dropna(subset=["age_bin_str"])
    if df2.empty:
        return _empty_fig("No fraud transactions in age bins")

    agg = df2.groupby("age_bin_str").agg(avg_fraud_amount=("amt", "mean")).reset_index()
    age_bin_order = df.attrs.get("age_bin_order", agg["age_bin_str"].tolist())
    agg["age_bin_str"] = pd.Categorical(agg["age_bin_str"], categories=age_bin_order, ordered=True)
    agg = agg.sort_values("age_bin_str")

    fig = px.bar(agg, x="age_bin_str", y="avg_fraud_amount", title="Average Fraud Amount per Age Bin", text="avg_fraud_amount", color_discrete_sequence=[_sample_color_from_scale(color)])
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(xaxis_tickangle=-45, height=420, margin=dict(t=40, b=80))
    return fig


def plot_category_age_heatmap(df: pd.DataFrame, category_col: str = "category", color: str = DEFAULT_COLOR) -> go.Figure:
    """
    Heatmap: counts of transaction categories per age bin.
    """
    if category_col not in df.columns or "age_bin_str" not in df.columns:
        return _empty_fig("Required columns not found")
    pivot = df.groupby([category_col, "age_bin_str"]).size().reset_index(name="count")
    if pivot.empty:
        return _empty_fig("No data for category-age pivot")
    cat_pivot = pivot.pivot(index=category_col, columns="age_bin_str", values="count").fillna(0)
    age_bin_order = df.attrs.get("age_bin_order", cat_pivot.columns.tolist())
    cat_pivot = cat_pivot.reindex(columns=age_bin_order, fill_value=0)

    fig = px.imshow(cat_pivot, text_auto=True, aspect="auto", title="Transaction Category Distribution per Age Bin", color_continuous_scale=color)
    fig.update_layout(height=600, margin=dict(t=40, b=80))
    return fig


def plot_job_age_heatmap(df: pd.DataFrame, job_col: str = "job", color: str = DEFAULT_COLOR) -> go.Figure:
    """
    Heatmap: top job categories per age bin (counts).
    """
    if job_col not in df.columns or "age_bin_str" not in df.columns:
        return _empty_fig("Required columns not found")
    top_jobs = df[job_col].value_counts().nlargest(10).index.tolist()
    pivot = df[df[job_col].isin(top_jobs)].groupby([job_col, "age_bin_str"]).size().reset_index(name="count")
    if pivot.empty:
        return _empty_fig("No job-age data")

    job_pivot = pivot.pivot(index=job_col, columns="age_bin_str", values="count").fillna(0)
    age_bin_order = df.attrs.get("age_bin_order", job_pivot.columns.tolist())
    job_pivot = job_pivot.reindex(columns=age_bin_order, fill_value=0)
    job_pivot = job_pivot.reindex(index=top_jobs, fill_value=0)

    fig = px.imshow(job_pivot, text_auto=True, aspect="auto", title="Top Job Categories per Age Bin", color_continuous_scale=color)
    fig.update_layout(height=600, margin=dict(t=40, b=80))
    return fig


def plot_top_job_per_age_bin_avg_amount(df: pd.DataFrame, color: str = DEFAULT_COLOR) -> go.Figure:
    """
    For each age bin, show the job with the highest average transaction amount.
    """
    required_cols = {"age_bin_str", "job", "amt"}
    if not required_cols.issubset(df.columns):
        return _empty_fig(f"Required columns missing: {required_cols - set(df.columns)}")

    # Aggregate
    agg_df = (
        df.groupby(["age_bin_str", "job"], observed=True)
        .agg(avg_amount=("amt", "mean"))
        .reset_index()
    )
    if agg_df.empty:
        return _empty_fig("No job-age-amount data")

    # Pick top job per age bin
    top_jobs = (
        agg_df.sort_values("avg_amount", ascending=False)
        .groupby("age_bin_str", observed=True)
        .head(1)
        .reset_index(drop=True)
    )

    # Apply ordering from df.attrs if available
    age_bin_order = [b for b in df.attrs.get("age_bin_order", top_jobs["age_bin_str"].tolist()) if b != "nan"]
    top_jobs["age_bin_str"] = pd.Categorical(top_jobs["age_bin_str"], categories=age_bin_order, ordered=True)
    top_jobs = top_jobs.sort_values("age_bin_str")

    # Plot — color by job for visual clarity
    fig = px.bar(
        top_jobs,
        x="age_bin_str",
        y="avg_amount",
        text="job",
        title="Top Job per Age Bin by Average Transaction Amount",
        color="job",
        color_discrete_sequence=px.colors.sequential.__getattribute__(color) if isinstance(color, str) else color
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-45, height=480, margin=dict(t=40, b=80))
    return fig



def plot_top_job_per_age_bin_count(df: pd.DataFrame, color: str = DEFAULT_COLOR) -> go.Figure:
    """
    For each age bin, show the job with the highest transaction count.
    """
    required_cols = {"age_bin_str", "job", "amt"}
    if not required_cols.issubset(df.columns):
        return _empty_fig(f"Required columns missing: {required_cols - set(df.columns)}")

    # Aggregate
    agg_df = (
        df.groupby(["age_bin_str", "job"], observed=True)
        .agg(transaction_count=("amt", "count"))
        .reset_index()
    )
    if agg_df.empty:
        return _empty_fig("No job-age-count data")

    # Pick top job per age bin
    top_jobs = (
        agg_df.sort_values("transaction_count", ascending=False)
        .groupby("age_bin_str", observed=True)
        .head(1)
        .reset_index(drop=True)
    )

    # Apply ordering from df.attrs if available
    age_bin_order = [b for b in df.attrs.get("age_bin_order", top_jobs["age_bin_str"].tolist()) if b != "nan"]
    top_jobs["age_bin_str"] = pd.Categorical(top_jobs["age_bin_str"], categories=age_bin_order, ordered=True)
    top_jobs = top_jobs.sort_values("age_bin_str")

    # Plot
    fig = px.bar(
        top_jobs,
        x="age_bin_str",
        y="transaction_count",
        text="job",  # Shows job label above bars
        title="Top Job per Age Bin by Transaction Count",
        color="job",  # Different colors per job
        color_discrete_sequence=px.colors.sequential.__getattribute__(color) if isinstance(color, str) else color
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-45, height=480, margin=dict(t=40, b=80))
    return fig




def plot_monthly_trends(df: pd.DataFrame, color: str = DEFAULT_COLOR) -> go.Figure:
    """
    Line + filled area showing monthly transaction counts.
    """
    if "year" not in df.columns or "month" not in df.columns:
        return _empty_fig("Required columns not found")
    monthly_stats = df.groupby(["year", "month"]).agg(total_amount=("amt", "sum"), transaction_count=("amt", "count")).reset_index()
    if monthly_stats.empty:
        return _empty_fig("No monthly data")
    monthly_stats["year_month"] = pd.to_datetime(monthly_stats[["year", "month"]].assign(day=1))

    base_color = _sample_color_from_scale(color)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_stats["year_month"], y=monthly_stats["transaction_count"], mode="lines+markers", name="Transaction Count", line=dict(color=base_color)))
    fig.add_trace(go.Scatter(x=monthly_stats["year_month"], y=monthly_stats["transaction_count"], mode="lines", fill="tozeroy", opacity=0.2, showlegend=False, line=dict(color=base_color)))
    fig.update_layout(title="Monthly Transaction Trends", xaxis_title="Month", yaxis_title="Transaction Count", height=480, margin=dict(t=40, b=80))
    return fig


def plot_avg_spend_by_hour(df: pd.DataFrame, color: str = DEFAULT_COLOR) -> go.Figure:
    """
    Line chart of average transaction amount by hour of day.
    """
    if "hour" not in df.columns or "amt" not in df.columns:
        return _empty_fig("Required columns not found")
    hour_avg_spend = df.groupby("hour")["amt"].mean().reset_index()
    if hour_avg_spend.empty:
        return _empty_fig("No hourly spend data")

    base_color = _sample_color_from_scale(color)
    fig = px.line(hour_avg_spend, x="hour", y="amt", markers=True, title="Average Spend by Hour of Day")
    fig.update_traces(line=dict(color=base_color))
    fig.update_xaxes(dtick=1, tick0=0)
    fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Average Transaction Amount", height=420, margin=dict(t=40, b=80))
    return fig


def plot_fraud_rate_over_time(df: pd.DataFrame, target: str = "is_fraud", color: str = DEFAULT_COLOR) -> go.Figure:
    """
    Line chart showing fraud rate (%) per month.
    """
    if "year" not in df.columns or "month" not in df.columns or target not in df.columns:
        return _empty_fig("Required columns not found")
    fraud_rate = df.groupby(["year", "month"]).agg(fraud_cases=(target, "sum"), total_transactions=(target, "count")).reset_index()
    if fraud_rate.empty:
        return _empty_fig("No fraud rate data")
    fraud_rate["fraud_rate_pct"] = fraud_rate["fraud_cases"] / fraud_rate["total_transactions"] * 100
    fraud_rate["year_month"] = pd.to_datetime(fraud_rate[["year", "month"]].assign(day=1))

    fig = px.line(fraud_rate, x="year_month", y="fraud_rate_pct", markers=True, title="Fraud Rate Over Time (%)", color_discrete_sequence=[_sample_color_from_scale(color)])
    fig.update_layout(yaxis_title="Fraud Rate (%)", xaxis_title="Month", height=480, margin=dict(t=40, b=80))
    return fig


def plot_weekday_fraud_bar(df: pd.DataFrame, target: str = "is_fraud", color: str = "Purples") -> go.Figure:
    if "trans_date_trans_time" in df.columns:
        ts = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        df = df.assign(
            weekday=ts.dt.day_name(),
            hour=ts.dt.hour
        )

    if "weekday" not in df.columns or target not in df.columns:
        return _empty_fig("Required columns not found")

    fraud_df = df[df[target] == 1]
    weekday_fraud = fraud_df.groupby("weekday")["amt"].count().reset_index(name="fraud_count")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_fraud["weekday"] = pd.Categorical(weekday_fraud["weekday"], categories=order, ordered=True)
    weekday_fraud = weekday_fraud.sort_values("weekday")

    if weekday_fraud.empty:
        return _empty_fig("No fraudulent transactions")

    fig = px.bar(
        weekday_fraud,
        x="weekday",
        y="fraud_count",
        title="Fraudulent Transactions by Weekday",
        color="fraud_count",
        color_continuous_scale=getattr(px.colors.sequential, color) if isinstance(color, str) else color
    )
    fig.update_layout(height=420, margin=dict(t=40, b=80))
    return fig


def plot_fraud_rate_heatmap_weekday_hour(df: pd.DataFrame, target: str = "is_fraud", color: str = "Purples") -> go.Figure:
    if "trans_date_trans_time" in df.columns:
        ts = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        df = df.assign(
            weekday=ts.dt.day_name(),
            hour=ts.dt.hour
        )

    if "weekday" not in df.columns or "hour" not in df.columns or target not in df.columns:
        return _empty_fig("Required columns not found")

    fraud_hourly = df.groupby(["weekday", "hour"])[target].mean().reset_index(name="fraud_rate")
    if fraud_hourly.empty:
        return _empty_fig("No fraud hourly data")

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = fraud_hourly.pivot(index="weekday", columns="hour", values="fraud_rate").reindex(order).fillna(0)

    fig = px.imshow(
        pivot * 100,
        text_auto=".1f",
        aspect="auto",
        title="Fraud Rate by Weekday & Hour (%)",
        origin="lower",
        color_continuous_scale=color
    )
    fig.update_layout(height=520, margin=dict(t=40, b=80))
    return fig



def plot_hourly_trends_normal_and_fraud(df: pd.DataFrame, target: str = "is_fraud", color: str = "Purples") -> go.Figure:
    """
    Side-by-side subplots: normal transactions (line) and fraudulent (bar) by hour.
    """
    if "hour" not in df.columns or target not in df.columns:
        return _empty_fig("Required columns not found")

    hourly_normal = df[df[target] == 0].groupby("hour")["amt"].count().reset_index(name="transaction_count")
    hourly_fraud = df[df[target] == 1].groupby("hour")["amt"].count().reset_index(name="transaction_count")

    base_color = _sample_color_from_scale(color)
    alt_color = _sample_color_from_scale(color, prefer_index=0)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Normal Transactions by Hour", "Fraudulent Transactions by Hour")
    )
    fig.add_trace(
        go.Scatter(x=hourly_normal["hour"], y=hourly_normal["transaction_count"], mode="lines+markers", name="Normal", line=dict(color=base_color)),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=hourly_fraud["hour"], y=hourly_fraud["transaction_count"], name="Fraud", marker_color=alt_color),
        row=1, col=2
    )

    fig.update_xaxes(tickmode="linear", tick0=0, dtick=1)
    fig.update_layout(height=420, showlegend=False, margin=dict(t=40, b=40))
    return fig


import plotly.express as px

def get_palette(name="Set2"):
    palettes = {
        "Set2": px.colors.qualitative.Set2,
        "Pastel": px.colors.qualitative.Pastel,
        "Bold": px.colors.qualitative.Bold,
    }
    return palettes.get(name, px.colors.qualitative.Set2)

def bin_ages(df, bins=None):
    bins = bins or [18, 25, 35, 45, 55, 65, 100]
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
    df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    return df

# Example usage in plotting functions:
def plot_age_distribution(df):
    df = bin_ages(df)
    fig = px.histogram(df, x="age_bin", color="is_fraud", barmode="group", color_discrete_sequence=get_palette())
    fig.update_layout(title="Age Distribution")
    return fig

