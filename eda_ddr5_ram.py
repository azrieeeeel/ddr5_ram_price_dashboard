import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================================================
# 1. Load data
# ===========================================================
df = pd.read_csv('costinflation-ddr5-ram-retail-prices-clean.csv')
df['observed_date'] = pd.to_datetime(df['observed_date'])

df['price_per_gb'] = df['price_amount'] / df['quantity_value']
df['brand'] = df['product_name'].str.extract(r'^([A-Za-z0-9.\-]+)')
df['brand'] = df['brand'].fillna('Unknown')

print('=== STATISTIK DESKRIPTIF ===')
print(df[['quantity_value', 'price_amount', 'normalized_price_amount', 'price_per_gb']].describe().T)

# ===========================================================
# 2. Analisis wilayah
# ===========================================================
geo_summary = (
    df.groupby('geography_label', as_index=False)
      .agg(
          avg_price_per_gb=('price_per_gb', 'mean'),
          median_price_per_gb=('price_per_gb', 'median'),
          min_price_per_gb=('price_per_gb', 'min'),
          max_price_per_gb=('price_per_gb', 'max'),
          total_produk=('product_name', 'count')
      )
      .sort_values('avg_price_per_gb', ascending=False)
)
print('\n=== RANGKING WILAYAH BERDASARKAN HARGA PER GB ===')
print(geo_summary.head(10).to_string(index=False))

# ===========================================================
# 3. Analisis kapasitas
# ===========================================================
capacity_summary = (
    df.groupby('quantity_value', as_index=False)
      .agg(
          avg_price_per_gb=('price_per_gb', 'mean'),
          avg_price=('price_amount', 'mean'),
          total_produk=('product_name', 'count')
      )
      .sort_values('quantity_value')
)
print('\n=== RANGKING KAPASITAS BERDASARKAN HARGA PER GB ===')
print(capacity_summary.head(10).to_string(index=False))

# ===========================================================
# 4. Analisis brand
# ===========================================================
brand_summary = (
    df.groupby('brand', as_index=False)
      .agg(
          avg_price_per_gb=('price_per_gb', 'mean'),
          median_price_per_gb=('price_per_gb', 'median'),
          total_produk=('product_name', 'count')
      )
      .sort_values('avg_price_per_gb')
)
print('\n=== BRAND DENGAN NILAI TERBAIK (HARGA PER GB PALING RENDAH) ===')
print(brand_summary.head(10).to_string(index=False))

# ===========================================================
# 5. Tren waktu
# ===========================================================
daily_trend = (
    df.groupby('observed_date', as_index=False)
      .agg(avg_price_per_gb=('price_per_gb', 'mean'), avg_price=('price_amount', 'mean'))
      .sort_values('observed_date')
)
print('\n=== TREND HARGA PER HARI ===')
print(daily_trend.head().to_string(index=False))

# ===========================================================
# 6. Korelasi antar variabel numerik
# ===========================================================
corr = df[['quantity_value', 'price_amount', 'normalized_price_amount', 'normalized_quantity_value', 'price_per_gb']].corr()
print('\n=== KORELASI ANTAR VARIABEL ===')
print(corr.round(3).to_string())

# ===========================================================
# 7. Visualisasi pola awal
# ===========================================================
sns.set_style('whitegrid')

# Wilayah: boxplot harga per GB
if not geo_summary.empty:
    top_geo = geo_summary.nlargest(10, 'avg_price_per_gb')['geography_label'].tolist()
    df_geo_top = df[df['geography_label'].isin(top_geo)]
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_geo_top, x='geography_label', y='price_per_gb')
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribusi Harga per GB per Wilayah (Top 10)')
    plt.ylabel('Harga per GB (USD)')
    plt.xlabel('Wilayah')
    plt.tight_layout()
    plt.savefig('eda_wilayah.png', dpi=150)
    plt.close()

# Trend harga per GB
plt.figure(figsize=(12, 5))
sns.lineplot(data=daily_trend, x='observed_date', y='avg_price_per_gb', marker='o')
plt.title('Tren Harga per GB DDR5 dari Waktu ke Waktu')
plt.xlabel('Tanggal')
plt.ylabel('Harga per GB (USD)')
plt.tight_layout()
plt.savefig('eda_trend.png', dpi=150)
plt.close()

# Hubungan kapasitas vs harga per GB
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='quantity_value', y='price_per_gb')
plt.title('Hubungan Kapasitas RAM vs Harga per GB')
plt.xlabel('Kapasitas (GB)')
plt.ylabel('Harga per GB (USD)')
plt.tight_layout()
plt.savefig('eda_kapasitas.png', dpi=150)
plt.close()

# Brand benchmarking
if not brand_summary.empty:
    top_brand = brand_summary.nlargest(10, 'total_produk').sort_values('avg_price_per_gb')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_brand, x='brand', y='avg_price_per_gb')
    plt.xticks(rotation=45, ha='right')
    plt.title('Rata-rata Harga per GB per Brand')
    plt.ylabel('Harga per GB (USD)')
    plt.xlabel('Brand')
    plt.tight_layout()
    plt.savefig('eda_brand.png', dpi=150)
    plt.close()

# Heatmap korelasi
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap Korelasi Variabel Numerik')
plt.tight_layout()
plt.savefig('eda_corr.png', dpi=150)
plt.close()

# ===========================================================
# 8. Insight awal
# ===========================================================
print('\n=== INSIGHT AWAL ===')
print('1) Wilayah dengan rata-rata harga per GB paling tinggi:', geo_summary.iloc[0]['geography_label'], '=', round(geo_summary.iloc[0]['avg_price_per_gb'], 2))
print('2) Wilayah dengan rata-rata harga per GB paling rendah:', geo_summary.iloc[-1]['geography_label'], '=', round(geo_summary.iloc[-1]['avg_price_per_gb'], 2))
print('3) Kapasitas paling efisien (harga per GB terendah):', capacity_summary.iloc[0]['quantity_value'], 'GB', 'dengan rata-rata', round(capacity_summary.iloc[0]['avg_price_per_gb'], 2))
print('4) Brand dengan rata-rata harga per GB paling rendah:', brand_summary.iloc[0]['brand'], '=', round(brand_summary.iloc[0]['avg_price_per_gb'], 2))
print('5) Korelasi paling kuat antara price_amount dan quantity_value:', round(corr.loc['price_amount', 'quantity_value'], 3))

print('\nFile visualisasi disimpan: eda_wilayah.png, eda_trend.png, eda_kapasitas.png, eda_brand.png, eda_corr.png')
