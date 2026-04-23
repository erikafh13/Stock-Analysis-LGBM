"""
pages/new_product_analysis.py
Halaman Analisis Produk Baru — produk yang ada di items.xlsx
tapi belum pernah muncul di data penjualan historis.

Logika utama:
1. Identifikasi: SKU di items.xlsx yang tidak ada di df_penjualan
2. Kategori ABC: mode kategori produk serupa (Kategori + Brand) dari hasil V2
   → Fallback: cari referensi dulu, kalau tidak ada → default "D"
3. SO WMA awal: rata-rata SO WMA produk serupa (Kategori + Brand + City) dari V2
   → Fallback: 0
4. Min/Max/Add Stock & Suggested PO: pakai fungsi V2 yang sama
5. Output: tabel per kota + pivot gabungan + dashboard + download
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from scipy import stats as scipy_stats

from utils.analysis import (
    DAYS_MULTIPLIER,
    MAX_MULTIPLIER,
    CITY_PREFIX_MAP,
    calculate_add_stock_v2,
    calculate_suggested_po_v2,
    calculate_all_summary_v2,
    calculate_persentase_stock,
    get_status_stock,
    melt_stock_by_city,
    highlight_kategori_abc_log,
    highlight_status_stock,
    convert_df_to_excel,
)

KAT_COL   = "Kategori ABC (Log-Benchmark - WMA)"
KEYS      = ["No. Barang", "Kategori Barang", "BRAND Barang", "Nama Barang"]
ALL_CITIES = ["SURABAYA", "JAKARTA", "SEMARANG", "JOGJA", "MALANG", "BALI"]


# ── Render Utama ───────────────────────────────────────────────────────────────
def render():
    st.title("🆕 Analisis Produk Baru")
    st.info("""
    **Logika halaman ini:**
    - ✅ Produk baru = SKU di file items yang **belum pernah ada** di data penjualan
    - ✅ Kategori ABC ditentukan dari **mode kategori produk serupa** (Kategori + Brand) di hasil V2
    - ✅ SO WMA awal dari **rata-rata SO WMA produk serupa** per kota dari hasil V2
    - ✅ Min/Max/Add Stock & Suggested PO menggunakan **logika V2 yang sama**
    - ⚠️ Halaman ini membutuhkan hasil **Analisa Stock V2** dan **Data Stock** sudah dimuat
    """)

    # ── Guard: pastikan V2 sudah jalan ────────────────────────────────────────
    v2_result = st.session_state.get("stock_v2_result")
    df_stock  = st.session_state.get("df_stock", pd.DataFrame())

    if v2_result is None or v2_result.empty:
        st.warning("⚠️ Harap jalankan **Hasil Analisa Stock V2** terlebih dahulu sebelum menggunakan halaman ini.")
        st.stop()

    if df_stock.empty:
        st.warning("⚠️ Data Stock belum dimuat. Harap muat di halaman **Input Data**.")
        st.stop()

    # ── Upload File Items ──────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📂 Upload File Items (Daftar Produk)")
    uploaded = st.file_uploader(
        "Upload file items (.xlsx) — daftar semua produk dari sistem",
        type=["xlsx"],
        key="items_uploader",
    )

    if uploaded is not None:
        try:
            df_items = pd.read_excel(uploaded)
            st.session_state["items_df"] = df_items
            st.success(f"✅ File berhasil dimuat: {len(df_items):,} baris.")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()

    items_df = st.session_state.get("items_df", pd.DataFrame())
    if items_df.empty:
        st.info("Silakan upload file items di atas untuk memulai.")
        st.stop()

    # Preview items
    with st.expander("👁️ Preview File Items"):
        st.dataframe(items_df.head(10), use_container_width=True)

    # ── Tombol Analisis ────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🚀 Jalankan Analisis Produk Baru"):
        _run_new_product_analysis(items_df, v2_result, df_stock)

    if st.session_state.get("new_product_result") is not None:
        _render_results()


# ── Kalkulasi Utama ────────────────────────────────────────────────────────────
def _run_new_product_analysis(items_df: pd.DataFrame, v2_result: pd.DataFrame, df_stock: pd.DataFrame):
    with st.spinner("Mengidentifikasi dan menganalisis produk baru..."):

        # ── STEP 1: Normalisasi kolom items ───────────────────────────────────
        items_df = items_df.copy()

        # Deteksi kolom SKU/No.Barang
        sku_col = None
        for candidate in ["SKU", "No. Barang", "No Barang", "Kode Barang", "Item Code"]:
            if candidate in items_df.columns:
                sku_col = candidate
                break
        if sku_col is None:
            st.error("❌ Kolom SKU/No. Barang tidak ditemukan di file items. Pastikan ada kolom bernama 'SKU' atau 'No. Barang'.")
            st.stop()

        # Deteksi kolom nama produk
        nama_col = None
        for candidate in ["Nama Accurate", "Nama Barang", "Keterangan Barang", "Nama", "Description"]:
            if candidate in items_df.columns:
                nama_col = candidate
                break

        # Deteksi kolom kategori & brand
        kat_col_items   = next((c for c in ["Kategori", "Kategori Barang", "Category"] if c in items_df.columns), None)
        brand_col_items = next((c for c in ["Brand", "BRAND", "BRAND Barang", "Merk"] if c in items_df.columns), None)

        items_df["_sku_norm"] = items_df[sku_col].astype(str).str.strip().str.upper()

        # ── STEP 2: Identifikasi produk baru ──────────────────────────────────
        df_penjualan = st.session_state.get("df_penjualan", pd.DataFrame())
        if not df_penjualan.empty and "No. Barang" in df_penjualan.columns:
            sold_skus = set(df_penjualan["No. Barang"].astype(str).str.strip().str.upper())
        else:
            # Fallback: pakai SKU dari v2_result
            sold_skus = set(v2_result["No. Barang"].astype(str).str.strip().str.upper())

        new_mask   = ~items_df["_sku_norm"].isin(sold_skus)
        df_new     = items_df[new_mask].copy()

        # Hapus baris kosong / opening balance
        if nama_col:
            df_new = df_new[df_new[nama_col].astype(str).str.strip() != ""]
            df_new = df_new[~df_new[nama_col].astype(str).str.upper().str.contains("OPENING BALANCE", na=False)]

        # Hapus "NO KATEGORI" / "NO BRAND"
        if kat_col_items:
            df_new = df_new[~df_new[kat_col_items].astype(str).str.upper().isin(["NO KATEGORI", ""])]
        if brand_col_items:
            df_new = df_new[~df_new[brand_col_items].astype(str).str.upper().isin(["NO BRAND", ""])]

        df_new.drop_duplicates(subset=["_sku_norm"], inplace=True)
        df_new.reset_index(drop=True, inplace=True)

        total_new = len(df_new)
        st.info(f"🆕 Ditemukan **{total_new} produk baru** (ada di items, belum pernah terjual).")

        if total_new == 0:
            st.success("✅ Tidak ada produk baru yang perlu dianalisis.")
            st.session_state["new_product_result"] = None
            return

        # ── STEP 3: Bangun referensi dari V2 ─────────────────────────────────
        # Normalisasi kolom referensi v2
        v2 = v2_result.copy()
        v2["_kat_norm"]   = v2["Kategori Barang"].astype(str).str.strip().str.upper()
        v2["_brand_norm"] = v2["BRAND Barang"].astype(str).str.strip().str.upper()

        # ABC mode per (Kategori, Brand, City)
        abc_ref = (
            v2.groupby(["_kat_norm", "_brand_norm", "City"])[KAT_COL]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "D")
            .reset_index()
            .rename(columns={KAT_COL: "ABC_Ref"})
        )

        # ABC mode per (Kategori, Brand) — fallback tanpa City
        abc_ref_noCity = (
            v2.groupby(["_kat_norm", "_brand_norm"])[KAT_COL]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "D")
            .reset_index()
            .rename(columns={KAT_COL: "ABC_Ref_noCity"})
        )

        # SO WMA rata-rata per (Kategori, Brand, City)
        so_ref = (
            v2[v2["SO WMA"] > 0]
            .groupby(["_kat_norm", "_brand_norm", "City"])["SO WMA"]
            .mean()
            .reset_index()
            .rename(columns={"SO WMA": "SO_Ref"})
        )

        # ── STEP 4: Buat baris per produk baru × City ─────────────────────────
        records = []
        for _, row in df_new.iterrows():
            sku      = row["_sku_norm"]
            nama     = str(row[nama_col]).strip() if nama_col else sku
            kategori = str(row[kat_col_items]).strip().upper() if kat_col_items else "UNKNOWN"
            brand    = str(row[brand_col_items]).strip().upper() if brand_col_items else "UNKNOWN"

            for city in ALL_CITIES:
                # ABC Ref: coba per city dulu
                abc_row = abc_ref[
                    (abc_ref["_kat_norm"] == kategori) &
                    (abc_ref["_brand_norm"] == brand) &
                    (abc_ref["City"] == city)
                ]
                if not abc_row.empty:
                    abc_val = abc_row.iloc[0]["ABC_Ref"]
                else:
                    # Fallback: tanpa city filter
                    abc_row2 = abc_ref_noCity[
                        (abc_ref_noCity["_kat_norm"] == kategori) &
                        (abc_ref_noCity["_brand_norm"] == brand)
                    ]
                    abc_val = abc_row2.iloc[0]["ABC_Ref_noCity"] if not abc_row2.empty else "D"

                # SO Ref: coba per city
                so_row = so_ref[
                    (so_ref["_kat_norm"] == kategori) &
                    (so_ref["_brand_norm"] == brand) &
                    (so_ref["City"] == city)
                ]
                so_val = float(so_row.iloc[0]["SO_Ref"]) if not so_row.empty else 0.0
                so_val = math.ceil(so_val)

                records.append({
                    "No. Barang":      sku,
                    "Kategori Barang": kategori,
                    "BRAND Barang":    brand,
                    "Nama Barang":     nama,
                    "City":            city,
                    KAT_COL:          abc_val,
                    "SO WMA":          so_val,
                    "SO_Referensi":    "Ada" if so_val > 0 else "Tidak Ada (Default D)",
                    "ABC_Referensi":   "Ada" if not abc_row.empty or not abc_row2.empty else "Tidak Ada (Default D)",
                })

        df_full = pd.DataFrame(records)

        # ── STEP 5: Gabungkan stock dari df_stock ─────────────────────────────
        stock_melted = melt_stock_by_city(df_stock)
        stock_melted["No. Barang"] = stock_melted["No. Barang"].astype(str).str.strip().str.upper()
        stock_melted["City"]       = stock_melted["City"].str.strip().str.upper()

        df_full = pd.merge(
            df_full,
            stock_melted[["No. Barang", "City", "Stock"]],
            on=["No. Barang", "City"],
            how="left",
        )
        df_full.rename(columns={"Stock": "Stock Cabang"}, inplace=True)
        df_full["Stock Cabang"] = df_full["Stock Cabang"].fillna(0).astype(int)

        # Stock Surabaya untuk referensi skenario PO
        sby_stock = df_full[df_full["City"] == "SURABAYA"][["No. Barang", "Stock Cabang"]].copy()
        sby_stock.rename(columns={"Stock Cabang": "Stock Surabaya"}, inplace=True)
        df_full = pd.merge(df_full, sby_stock, on="No. Barang", how="left")
        df_full["Stock Surabaya"] = df_full["Stock Surabaya"].fillna(0).astype(int)

        # ── STEP 6: Hitung Min / Max / Add Stock ──────────────────────────────
        # Min Stock = ceil(SO WMA × Days Multiplier)
        mult_days = df_full[KAT_COL].map(DAYS_MULTIPLIER).fillna(1.0)
        df_full["Min Stock"] = np.where(
            (df_full["SO WMA"] <= 0) | (df_full[KAT_COL] == "F"),
            0,
            np.ceil(df_full["SO WMA"] * mult_days),
        ).astype(int)

        mult_max = df_full[KAT_COL].map(MAX_MULTIPLIER).fillna(1.0)
        df_full["Max Stock"] = np.where(
            df_full[KAT_COL] == "F",
            1,
            np.ceil(df_full["SO WMA"] * mult_max),
        ).astype(int)

        # Add Stock V2 (dengan bonus A/B)
        df_full["Add Stock"] = calculate_add_stock_v2(
            df_full, KAT_COL, "SO WMA", "Stock Cabang"
        )

        # Persentase Stock
        df_full["Persentase Stock"] = np.where(
            df_full["Min Stock"] > 0,
            (df_full["Stock Cabang"] / df_full["Min Stock"]) * 100,
            np.where(df_full["Stock Cabang"] > 0, 10000, 0),
        ).round(1)

        # Status Stock
        df_full["Status Stock"] = df_full.apply(get_status_stock, axis=1)

        # ── STEP 7: Suggested PO (3 skenario V2) ──────────────────────────────
        df_full["Suggested PO"] = calculate_suggested_po_v2(df_full)

        # Sisa Stock Surabaya
        sby_sisa = df_full[df_full["City"] == "SURABAYA"][["No. Barang", "Stock Cabang", "Min Stock"]].copy()
        sby_sisa["Sisa Stock Surabaya"] = np.maximum(0, sby_sisa["Stock Cabang"] - sby_sisa["Min Stock"])
        df_full = pd.merge(
            df_full,
            sby_sisa[["No. Barang", "Sisa Stock Surabaya"]],
            on="No. Barang",
            how="left",
        )
        df_full["Sisa Stock Surabaya"] = df_full["Sisa Stock Surabaya"].fillna(0).astype(int)

        # Bulatkan semua int cols
        int_cols = ["Stock Cabang", "Min Stock", "Max Stock", "Add Stock", "Suggested PO", "SO WMA"]
        for col in int_cols:
            if col in df_full.columns:
                df_full[col] = pd.to_numeric(df_full[col], errors="coerce").fillna(0).round(0).astype(int)

        st.session_state["new_product_result"] = df_full.copy()
        st.success(f"✅ Analisis produk baru selesai! {total_new} produk × {len(ALL_CITIES)} kota.")


# ── Render Hasil ───────────────────────────────────────────────────────────────
def _render_results():
    result = st.session_state["new_product_result"].copy()
    result = result[result["City"] != "OTHERS"]

    st.markdown("---")
    st.header("Filter Produk Baru")
    col_f1, col_f2, col_f3 = st.columns(3)
    sel_kat   = col_f1.multiselect("Kategori:", sorted(result["Kategori Barang"].dropna().unique().astype(str)), key="np_kat")
    sel_brand = col_f2.multiselect("Brand:",    sorted(result["BRAND Barang"].dropna().unique().astype(str)),    key="np_brand")
    sel_abc   = col_f3.multiselect("Kategori ABC:", sorted(result[KAT_COL].dropna().unique().astype(str)),       key="np_abc")

    if sel_kat:   result = result[result["Kategori Barang"].astype(str).isin(sel_kat)]
    if sel_brand: result = result[result["BRAND Barang"].astype(str).isin(sel_brand)]
    if sel_abc:   result = result[result[KAT_COL].astype(str).isin(sel_abc)]

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📍 Tabel per Kota", "📊 Tabel Gabungan", "📈 Dashboard"])

    with tab1:
        _render_per_kota(result)
    with tab2:
        _render_pivot(result)
    with tab3:
        _render_dashboard(result)

    # ── Download ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("💾 Unduh Hasil Analisis Produk Baru")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="Semua Kota", index=False)
        for city in sorted(result["City"].unique()):
            city_df = result[result["City"] == city]
            if not city_df.empty:
                city_df.to_excel(writer, sheet_name=city[:31], index=False)
        # Summary
        summary = _build_summary(result)
        if not summary.empty:
            summary.to_excel(writer, sheet_name="Summary", index=False)

    st.download_button(
        "📥 Unduh Excel Produk Baru",
        data=output.getvalue(),
        file_name="Analisis_Produk_Baru.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ── Tabel per Kota ─────────────────────────────────────────────────────────────
def _render_per_kota(result: pd.DataFrame):
    st.header("Hasil Analisis per Kota")
    header_style = {"selector": "th", "props": [("background-color", "#0068c9"),
                                                  ("color", "white"), ("text-align", "center")]}
    display_order = (
        KEYS
        + ["City", KAT_COL, "ABC_Referensi", "SO_Referensi",
           "SO WMA", "Stock Cabang", "Min Stock", "Max Stock",
           "Persentase Stock", "Status Stock", "Add Stock", "Suggested PO", "Sisa Stock Surabaya"]
    )

    for city in sorted(result["City"].unique()):
        city_df = result[result["City"] == city].copy()
        if city_df.empty:
            continue
        with st.expander(f"📍 {city} — {len(city_df)} produk baru"):
            disp_cols = [c for c in display_order if c in city_df.columns and c != "City"]
            city_disp = city_df[disp_cols]

            fmt = {}
            for col in city_disp.columns:
                if col in KEYS or col in ["ABC_Referensi", "SO_Referensi", "Status Stock", KAT_COL]:
                    continue
                if pd.api.types.is_numeric_dtype(city_disp[col]):
                    fmt[col] = "{:.1f}" if "Persentase" in col else "{:.0f}"

            st.dataframe(
                city_disp.style
                .format(fmt, na_rep="-")
                .apply(lambda x: x.map(highlight_kategori_abc_log), subset=[KAT_COL])
                .apply(lambda x: x.map(highlight_status_stock),     subset=["Status Stock"])
                .set_table_styles([header_style]),
                use_container_width=True,
            )


# ── Tabel Gabungan Pivot ───────────────────────────────────────────────────────
def _render_pivot(result: pd.DataFrame):
    st.header("📊 Tabel Gabungan Seluruh Kota")

    CITY_SHORT = {
        "BALI": "BALI", "JAKARTA": "JKT", "JOGJA": "JOG",
        "MALANG": "MLG", "SEMARANG": "SMG", "SURABAYA": "SBY",
    }
    pivot_vals = ["SO WMA", "Stock Cabang", "Min Stock", "Max Stock",
                  "Add Stock", "Suggested PO", "Persentase Stock", "Status Stock", KAT_COL]
    existing_vals = [c for c in pivot_vals if c in result.columns]

    pivot = result.pivot_table(index=KEYS, columns="City", values=existing_vals, aggfunc="first")
    pivot.columns = [f"{CITY_SHORT.get(city, city)}_{metric}" for metric, city in pivot.columns]
    pivot.reset_index(inplace=True)

    # Summary per SKU
    summary = _build_summary(result)
    if not summary.empty:
        pivot = pd.merge(pivot, summary, on=KEYS, how="left")

    # Tipe kolom
    num_cols = [c for c in pivot.columns if c not in KEYS and pd.api.types.is_numeric_dtype(pivot[c])]
    obj_cols = [c for c in pivot.columns if c not in KEYS and c not in num_cols]
    pivot[num_cols] = pivot[num_cols].fillna(0).astype(int)
    pivot[obj_cols] = pivot[obj_cols].fillna("-")

    col_cfg = {c: st.column_config.NumberColumn(format="%.0f") for c in num_cols}
    st.dataframe(pivot, column_config=col_cfg, use_container_width=True)

    # Legenda skenario
    st.markdown("---")
    st.markdown("**Legenda Skenario Distribusi:**")
    c1, c2, c3 = st.columns(3)
    c1.error("**1 - KURANG**\nStok Sisa SBY = 0.\nSemua cabang PO = 0.")
    c2.warning("**2 - TERBATAS**\nStok Sisa ada tapi < Total kebutuhan.\nDistribusi berdasarkan urgency.")
    c3.success("**3 - OVER**\nStok Sisa SBY cukup untuk semua cabang.")


# ── Summary per SKU ────────────────────────────────────────────────────────────
def _build_summary(result: pd.DataFrame) -> pd.DataFrame:
    """Hitung summary per No. Barang untuk pivot dan download."""
    rows = []
    for no_barang, grp in result.groupby("No. Barang"):
        mask_sby = grp["City"] == "SURABAYA"
        mask_cab = ~mask_sby
        sby_rows  = grp.loc[mask_sby]
        stock_sby = sby_rows["Stock Cabang"].sum()
        min_sby   = sby_rows["Min Stock"].sum()
        stok_sisa = max(0, stock_sby - min_sby)
        add_cabang = grp.loc[mask_cab, "Add Stock"].sum()
        add_sby    = sby_rows["Add Stock"].sum()
        need_sup   = max(0, add_cabang + add_sby)

        if stok_sisa <= 0:
            skenario = "1 - KURANG"
        elif stok_sisa >= add_cabang:
            skenario = "3 - OVER"
        else:
            skenario = "2 - TERBATAS"

        first = grp.iloc[0]
        row = {k: first[k] for k in KEYS if k in grp.columns}
        row.update({
            "All_Add_Stock_Cabang":   int(add_cabang),
            "All_Need_From_Supplier": int(need_sup),
            "All_Restock_1_Bulan":    "PO" if need_sup > 0 else "NO",
            "Skenario_Distribusi":    skenario,
            "All_Stock_Cabang":       int(grp["Stock Cabang"].sum()),
            "All_SO_WMA":             int(grp["SO WMA"].sum()),
        })
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Dashboard ──────────────────────────────────────────────────────────────────
def _render_dashboard(result: pd.DataFrame):
    st.header("📈 Dashboard Produk Baru")
    if result.empty:
        st.info("Tidak ada data untuk ditampilkan.")
        return

    # ── Kartu ringkasan ────────────────────────────────────────────────────────
    total_sku     = result["No. Barang"].nunique()
    ada_ref       = result[result["ABC_Referensi"] == "Ada"]["No. Barang"].nunique()
    tanpa_ref     = total_sku - ada_ref
    total_add     = result["Add Stock"].sum()
    perlu_po      = result[result["Add Stock"] > 0]["No. Barang"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🆕 Total SKU Baru",       f"{total_sku} SKU")
    c2.metric("✅ Ada Referensi V2",      f"{ada_ref} SKU")
    c3.metric("⚠️ Default D (No Ref.)",  f"{tanpa_ref} SKU")
    c4.metric("📦 Total Add Stock",       f"{int(total_add)} unit")

    st.markdown("---")

    # ── Distribusi ABC & Status ────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribusi Kategori ABC (Referensi)")
        # Dedup per SKU (ambil satu baris per produk — semua city harusnya sama)
        abc_dist = result.drop_duplicates("No. Barang")[KAT_COL].value_counts()
        st.bar_chart(abc_dist)

    with col_b:
        st.subheader("Distribusi Status Stock per City")
        status_dist = result["Status Stock"].value_counts()
        st.bar_chart(status_dist)

    st.markdown("---")

    # ── Top produk butuh Add Stock terbanyak ──────────────────────────────────
    summary = _build_summary(result)
    if not summary.empty:
        st.subheader("🔝 Top 10 Produk Baru — Add Stock Terbesar")
        top10 = summary.nlargest(10, "All_Add_Stock_Cabang")[
            ["Nama Barang", "Kategori Barang", "BRAND Barang",
             "All_Add_Stock_Cabang", "All_Need_From_Supplier",
             "All_Restock_1_Bulan", "Skenario_Distribusi"]
        ]
        st.dataframe(top10, use_container_width=True)

        st.markdown("---")
        col_s1, col_s2, col_s3 = st.columns(3)
        skenario_counts = summary["Skenario_Distribusi"].value_counts()
        col_s1.error(f"**1 - KURANG**\n{skenario_counts.get('1 - KURANG', 0)} SKU")
        col_s2.warning(f"**2 - TERBATAS**\n{skenario_counts.get('2 - TERBATAS', 0)} SKU")
        col_s3.success(f"**3 - OVER**\n{skenario_counts.get('3 - OVER', 0)} SKU")

    st.markdown("---")

    # ── Produk baru per kategori barang ───────────────────────────────────────
    st.subheader("📂 Jumlah Produk Baru per Kategori")
    kat_dist = result.drop_duplicates("No. Barang")["Kategori Barang"].value_counts().head(15)
    st.bar_chart(kat_dist)
