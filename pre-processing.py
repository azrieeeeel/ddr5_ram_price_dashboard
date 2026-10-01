import pandas as pd
import numpy as np

# ============================================
# 1. Load dataset
# ============================================
df = pd.read_csv(
    "costinflation-ddr5-ram-retail-prices-raw-2026-07-13-to-2026-08-10.csv"
)

print("Jumlah data awal:", len(df))
print("Kolom:", list(df.columns))
print("Missing value awal:\n", df.isna().sum())
print("Duplikasi awal:", df.duplicated().sum())

# ============================================================
# 2. Cek dan menangani missing value
# ============================================================
# Tujuan:
# - Nilai numerik diisi dengan median agar tidak terlalu sensitif
#   terhadap nilai ekstrem.
# - Nilai kategorikal diisi dengan mode (nilai yang paling sering)
#   atau label "Unknown" bila tidak ada mode.
# - Tanggal yang tidak valid diubah ke datetime, lalu diisi jika perlu.

categorical_cols = [
    "series_id", "series_title", "canonical_url", "geography_type",
    "geography_label", "product_name", "quantity_unit", "quantity_name",
    "currency_code", "normalized_quantity_unit"
]

numeric_cols = [
    "quantity_value", "price_amount", "normalized_price_amount",
    "normalized_quantity_value"
]

# Isi kolom kategorikal
for col in categorical_cols:
    if col in df.columns:
        if df[col].isna().any():
            mode_value = df[col].mode()
            fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)

# Isi kolom numerik
for col in numeric_cols:
    if col in df.columns:
        if df[col].isna().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

# Tangani kolom tanggal
if "observed_date" in df.columns:
    df["observed_date"] = pd.to_datetime(df["observed_date"], errors="coerce")

    # Jika masih ada tanggal yang kosong, isi dengan tanggal yang paling umum
    if df["observed_date"].isna().any():
        common_date = df["observed_date"].dropna().mode()
        if not common_date.empty:
            df["observed_date"] = df["observed_date"].fillna(common_date.iloc[0])
        else:
            df = df.dropna(subset=["observed_date"]).copy()

print("\nMissing value setelah penanganan:\n", df.isna().sum())

# ============================================================
# 3. Mendeteksi dan menghapus duplikasi data
# ============================================================
# Tujuan:
# - Membersihkan data yang sama persis berulang.
# - Duplikasi bisa muncul akibat scraping, export berulang, atau merge file.

df = df.drop_duplicates().reset_index(drop=True)

print("\nDuplikasi setelah pembersihan:", df.duplicated().sum())

# ============================================================
# 4. Mendeteksi dan menangani outlier pada kolom numerik
# ============================================================
# Tujuan:
# - Mengidentifikasi nilai yang sangat ekstrem dibanding distribusi normal.
# - Metode yang umum dipakai adalah IQR (Interquartile Range).
# - Alih-alih langsung membuang row, kita bisa "capping" (membatasi)
#   agar data tetap utuh tetapi outlier tidak terlalu mengganggu analisis.

for col in numeric_cols:
    if col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Capping outlier
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        # Optional: jika ingin menghapus row outlier, gunakan perintah berikut:
        # df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# ============================================================
# 5. Menyesuaikan tipe data
# ============================================================
# Tujuan:
# - Mengubah kolom ke format yang benar agar analisis lebih efisien
#   dan konsisten.

# Tipe kategorikal
category_cols = [
    "series_id", "series_title", "canonical_url", "geography_type",
    "geography_label", "product_name", "quantity_unit", "quantity_name",
    "currency_code", "normalized_quantity_unit"
]

for col in category_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

# geography_id biasanya identitas lokasi, jadi lebih tepat sebagai string/kategori
if "geography_id" in df.columns:
    df["geography_id"] = df["geography_id"].astype(str).astype("category")

# Tipe numerik eksplisit
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Tipe datetime
if "observed_date" in df.columns:
    df["observed_date"] = pd.to_datetime(df["observed_date"], errors="coerce")

# ============================================================
# 6. Ringkasan hasil final
# ============================================================
print("\nInfo tipe data akhir:")
print(df.dtypes)

print("\nJumlah data akhir:", len(df))
print("Missing value akhir:\n", df.isna().sum())
print("Duplikasi akhir:", df.duplicated().sum())

# Simpan hasil cleaning
df.to_csv("costinflation-ddr5-ram-retail-prices-clean.csv", index=False)
print("\nFile hasil cleaning disimpan: costinflation-ddr5-ram-retail-prices-clean.csv")