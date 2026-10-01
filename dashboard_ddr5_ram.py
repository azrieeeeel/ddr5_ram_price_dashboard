import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="DDR5 RAM Price Dashboard",
    page_icon="💾",
    layout="wide",
)

# -----------------------------------------------------
# Load data
# -----------------------------------------------------
@st.cache_data

def load_data():
    df = pd.read_csv("costinflation-ddr5-ram-retail-prices-clean.csv")
    df["observed_date"] = pd.to_datetime(df["observed_date"])
    df = df[df["quantity_value"] > 0].copy()
    df["price_per_gb"] = df["price_amount"] / df["quantity_value"]
    df["brand"] = df["product_name"].str.extract(r"^([A-Za-z0-9.\-]+)")
    df["brand"] = df["brand"].fillna("Unknown")
    return df


df = load_data()

# -----------------------------------------------------
# Sidebar filters
# -----------------------------------------------------
st.sidebar.header("Filter Eksplorasi")

min_date = df["observed_date"].min().date()
max_date = df["observed_date"].max().date()
start_date, end_date = st.sidebar.date_input(
    "Pilih rentang tanggal",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date,
)

if isinstance(start_date, tuple):
    start_date = start_date[0]
    end_date = start_date if end_date is None else end_date

# Jika date_input returns a single date, masukkan ke tuple
if not isinstance(start_date, tuple):
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

# geography filter
all_geography = df["geography_label"].dropna().unique().tolist()
selected_geography = st.sidebar.multiselect(
    "Pilih wilayah",
    options=all_geography,
    default=all_geography[:10],
)

# brand filter
all_brands = df["brand"].dropna().unique().tolist()
selected_brands = st.sidebar.multiselect(
    "Pilih brand",
    options=all_brands,
    default=all_brands[:10],
)

# capacity range filter
min_capacity = int(df["quantity_value"].min())
max_capacity = int(df["quantity_value"].max())
capacity_range = st.sidebar.slider(
    "Rentang kapasitas (GB)",
    min_value=min_capacity,
    max_value=max_capacity,
    value=(min_capacity, max_capacity),
)

filtered_df = df[
    (df["observed_date"].dt.date >= start_date)
    & (df["observed_date"].dt.date <= end_date)
    & (df["geography_label"].isin(selected_geography))
    & (df["brand"].isin(selected_brands))
    & (df["quantity_value"].between(capacity_range[0], capacity_range[1]))
].copy()

if filtered_df.empty:
    st.warning("Tidak ada data yang sesuai dengan filter yang dipilih. Silakan ubah filter pada sidebar.")
    st.stop()

# -----------------------------------------------------
# Prepare data summaries
# -----------------------------------------------------
geo_summary = (
    filtered_df.groupby("geography_label", as_index=False)
    .agg(avg_price_per_gb=("price_per_gb", "mean"), total_produk=("product_name", "count"))
    .sort_values("avg_price_per_gb", ascending=False)
    .head(10)
)

capacity_summary = (
    filtered_df.groupby("quantity_value", as_index=False)
    .agg(avg_price_per_gb=("price_per_gb", "mean"), avg_price=("price_amount", "mean"))
    .sort_values("quantity_value")
)

brand_summary = (
    filtered_df.groupby("brand", as_index=False)
    .agg(avg_price_per_gb=("price_per_gb", "mean"), total_produk=("product_name", "count"))
    .sort_values("avg_price_per_gb")
    .head(10)
)

industry_brand = (
    filtered_df["brand"].value_counts().head(6).reset_index()
)
industry_brand.columns = ["brand", "jumlah_produk"]

daily_trend = (
    filtered_df.groupby("observed_date", as_index=False)
    .agg(avg_price_per_gb=("price_per_gb", "mean"))
    .sort_values("observed_date")
)

corr_cols = ["quantity_value", "price_amount", "normalized_price_amount", "price_per_gb"]
corr = filtered_df[corr_cols].corr()

# -----------------------------------------------------
# Dashboard title and KPI cards
# -----------------------------------------------------
st.title("DDR5 RAM Price Dashboard")
st.caption("Eksplorasi harga, tren, dan kinerja brand berdasarkan wilayah, kapasitas, dan tanggal")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Rata-rata Harga / GB", f"${filtered_df['price_per_gb'].mean():.2f}")
with kpi2:
    st.metric("Rata-rata Harga Total", f"${filtered_df['price_amount'].mean():.2f}")
with kpi3:
    st.metric("Jumlah Produk", f"{len(filtered_df):,}")
with kpi4:
    st.metric("Wilayah Terlihat", f"{filtered_df['geography_label'].nunique()}")

# -----------------------------------------------------
# Chart 1: Bar chart harga per GB per wilayah
# -----------------------------------------------------
fig1 = px.bar(
    geo_summary,
    x="avg_price_per_gb",
    y="geography_label",
    orientation="h",
    color="avg_price_per_gb",
    color_continuous_scale="Viridis",
    title="Perbandingan Harga per GB per Wilayah (Top 10)",
    labels={"geography_label": "Wilayah", "avg_price_per_gb": "Harga per GB (USD)"},
    height=500,
)
fig1.update_layout(template="plotly_white")

# Chart 2: line chart trend
fig2 = px.line(
    daily_trend,
    x="observed_date",
    y="avg_price_per_gb",
    markers=True,
    title="Tren Harga per GB dari Waktu ke Waktu",
    labels={"observed_date": "Tanggal", "avg_price_per_gb": "Harga per GB (USD)"},
    color_discrete_sequence=["#1f77b4"],
    height=400,
)
fig2.update_layout(template="plotly_white")

# Chart 3: scatter plot kapasitas vs harga per GB
fig3 = px.scatter(
    filtered_df,
    x="quantity_value",
    y="price_per_gb",
    color="geography_type",
    hover_name="product_name",
    title="Hubungan Kapasitas RAM vs Harga per GB",
    labels={"quantity_value": "Kapasitas (GB)", "price_per_gb": "Harga per GB (USD)", "geography_type": "Tipe Wilayah"},
    height=500,
)
fig3.update_layout(template="plotly_white")

# Chart 4: box plot distribusi harga per GB by geography
box_df = filtered_df[filtered_df["geography_label"].isin(geo_summary["geography_label"].tolist())].copy()
fig4 = px.box(
    box_df,
    x="geography_label",
    y="price_per_gb",
    color="geography_label",
    title="Distribusi Harga per GB Antar Wilayah",
    labels={"geography_label": "Wilayah", "price_per_gb": "Harga per GB (USD)"},
    height=500,
)
fig4.update_layout(showlegend=False, template="plotly_white")
fig4.update_xaxes(tickangle=45)

# Chart 5: heatmap korelasi
fig5 = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale="RdBu_r",
    zmin=-1,
    zmax=1,
    hoverongaps=False,
    colorbar=dict(title="Korelasi")
))
fig5.update_layout(
    title="Heatmap Korelasi Variabel Numerik",
    template="plotly_white",
    height=500,
)

# Chart 6: pie chart brand distribution
fig6 = px.pie(
    industry_brand,
    names="brand",
    values="jumlah_produk",
    title="Proporsi Produk Berdasarkan Brand Utama",
    color_discrete_sequence=px.colors.qualitative.Set3,
    height=500,
)
fig6.update_traces(textposition="inside", textinfo="percent+label")
fig6.update_layout(template="plotly_white")

# -----------------------------------------------------
# Layout dashboard
# -----------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(fig3, use_container_width=True)
with col4:
    st.plotly_chart(fig4, use_container_width=True)

col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(fig5, use_container_width=True)
with col6:
    st.plotly_chart(fig6, use_container_width=True)

# Optional insight summary
st.subheader("Insight Singkat")
if not geo_summary.empty:
    st.write(f"- Wilayah paling mahal: {geo_summary.iloc[0]['geography_label']} dengan rata-rata ${geo_summary.iloc[0]['avg_price_per_gb']:.2f}/GB")
    st.write(f"- Wilayah paling efisien: {geo_summary.iloc[-1]['geography_label']} dengan rata-rata ${geo_summary.iloc[-1]['avg_price_per_gb']:.2f}/GB")
if not brand_summary.empty:
    st.write(f"- Brand dengan harga per GB terendah: {brand_summary.iloc[0]['brand']} (${brand_summary.iloc[0]['avg_price_per_gb']:.2f}/GB)")
