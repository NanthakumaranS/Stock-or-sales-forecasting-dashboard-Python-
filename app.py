import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import generate_sales, prepare
from forecasting import MODELS, backtest, forecast

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")
st.title("Sales Forecasting Dashboard")


@st.cache_data
def demo_data():
    return generate_sales()


with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Demo data", "Upload CSV"])
    if source == "Demo data":
        df = demo_data()
    else:
        file = st.file_uploader("CSV file", type="csv")
        if file is None:
            st.info("Upload a CSV with at least a date column and a sales column.")
            st.stop()
        raw = pd.read_csv(file)
        cols = list(raw.columns)
        date_col = st.selectbox("Date column", cols)
        sales_col = st.selectbox("Sales column", cols, index=min(1, len(cols) - 1))
        cat_col = st.selectbox("Category (optional)", ["None"] + cols)
        reg_col = st.selectbox("Region (optional)", ["None"] + cols)
        df = prepare(raw, date_col, sales_col,
                     None if cat_col == "None" else cat_col,
                     None if reg_col == "None" else reg_col)

    st.header("Filters")
    cats = st.multiselect("Category", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
    regs = st.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))

    st.header("Forecast")
    horizon = st.slider("Months ahead", 3, 24, 12)
    model = st.selectbox("Model", MODELS, index=1)
    drop_last = st.checkbox("Drop last (partial) month", value=False)

filtered = df[df["category"].isin(cats) & df["region"].isin(regs)]
if filtered.empty:
    st.warning("No data for the selected filters.")
    st.stop()

monthly = filtered.set_index("order_date").resample("MS")["sales"].sum()
if drop_last:
    monthly = monthly.iloc[:-1]
if len(monthly) < 24:
    st.error("Need at least 24 months of data to forecast with yearly seasonality.")
    st.stop()

# KPI cards
last12, prev12 = monthly.iloc[-12:].sum(), monthly.iloc[-24:-12].sum()
yoy = (last12 / prev12 - 1) * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total sales", f"{monthly.sum():,.0f}")
c2.metric("Avg monthly sales", f"{monthly.mean():,.0f}")
c3.metric("Last 12 mo vs prior 12", f"{yoy:+.1f}%")
c4.metric("Best month", monthly.idxmax().strftime("%b %Y"))

# Forecast chart
try:
    fc = forecast(monthly, horizon, model)
except Exception as e:
    st.error(f"Model failed: {e}")
    st.stop()

fig = go.Figure()
fig.add_trace(go.Scatter(x=monthly.index, y=monthly.values, name="Actual", line=dict(width=2)))
fig.add_trace(go.Scatter(x=fc.index, y=fc["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fc.index, y=fc["lower"], fill="tonexty", line=dict(width=0),
                         name="95% interval", fillcolor="rgba(255,127,14,0.2)", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fc.index, y=fc["yhat"], name=f"Forecast ({model})",
                         line=dict(dash="dash", width=2, color="#ff7f0e")))
fig.update_layout(title="Monthly sales and forecast", yaxis_title="Sales", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# Breakdowns
left, right = st.columns(2)
by_cat = filtered.groupby("category")["sales"].sum().reset_index()
left.plotly_chart(px.bar(by_cat, x="category", y="sales", title="Sales by category"), use_container_width=True)
season = monthly.groupby(monthly.index.month).mean().reset_index()
season.columns = ["month", "avg_sales"]
right.plotly_chart(px.bar(season, x="month", y="avg_sales", title="Average sales by calendar month"),
                   use_container_width=True)

# Model comparison
st.subheader("Model comparison (backtest on last 6 months)")
st.dataframe(backtest(monthly, 6).style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE %": "{:.1f}"}),
             use_container_width=True)

st.download_button("Download forecast CSV", fc.round(2).to_csv(index_label="month"), "forecast.csv")
