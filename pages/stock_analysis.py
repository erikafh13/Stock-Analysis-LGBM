"""
pages/stock_analysis.py
Halaman Hasil Analisa Stock — preprocessing, kalkulasi, tabel, dashboard.
"""

import math
import re
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from datetime import timedelta

from utils import (
    BULAN_INDONESIA,
    map_nama_dept,
    map_city,
    convert_df_to_excel,
    calculate_daily_wma,
    classify_abc_log_benchmark,
    calculate_min_stock,
    calculate_max_stock,
    calculate_add_stock,
    calculate_suggested_po,
    get_status_stock,
    melt_stock_by_city,
    highlight_kategori_abc_log,
    highlight_status_stock,
)


# ── Render Utama ───────────────────────────────────────────────────────────────
def render():
    st.title("📈 Hasil Analisa Stock")

    # Validasi data tersedia
    if (
        st.session_state.df_penjualan.empty
        or st.session_state.produk_ref.empty
        or st.session_state.df_stock.empty
    ):
        st.warning("⚠️ Harap muat semua file di halaman **'Input Data'** terlebih dahulu.")
        st.stop()

    # ── Preprocessing ──────────────────────────────────────────────────────────
    penjualan  = st.session_state.df_penjualan.copy()
    produk_ref = st.session_state.produk_ref.copy()
    df_stock   = st.session_state.df_stock.copy()

    for df in [penjualan, produk_ref, df_stock]:
        if "No. Barang" in df.columns:
            df["No. Barang"] = df["No. Barang"].astype(str).str.strip()

    # Deduplikasi faktur
    if "No. Faktur" in penjualan.columns and "No. Barang" in penjualan.columns:
        penjualan["No. Faktur"]     = penjualan["No. Faktur"].astype(str).str.strip()
        penjualan["Faktur + Barang"] = penjualan["No. Faktur"] + penjualan["No. Barang"]
        duplicates = penjualan[penjualan.duplicated(subset=["Faktur + Barang"], keep=False)]
        penjualan.drop_duplicates(subset=["Faktur + Barang"], keep="first", inplace=True)
        st.session_state.df_penjualan = penjualan.copy()
        if not duplicates.empty:
            deleted = duplicates[~duplicates.index.isin(penjualan.index)]
            st.warning(f"⚠️ Ditemukan dan dihapus {len(deleted)} baris duplikat 'Faktur + Barang'.")
            with st.expander("Lihat Detail Duplikat yang Dihapus"):
                st.dataframe(deleted)
        else:
            st.info("✅ Tidak ada duplikat 'Faktur + Barang' yang ditemukan.")

    penjualan.rename(columns={"Qty": "Kuantitas"}, inplace=True, errors="ignore")
    penjualan["Nama Dept"] = penjualan.apply(map_nama_dept, axis=1)
    penjualan["City"]      = penjualan["Nama Dept"].apply(map_city)

    produk_ref.rename(columns={"Keterangan Barang": "Nama Barang"}, inplace=True, errors="ignore")
    if "Kategori Barang" in produk_ref.columns:
        produk_ref["Kategori Barang"] = produk_ref["Kategori Barang"].astype(str).str.strip().str.upper()
    if "City" in penjualan.columns:
        penjualan["City"] = penjualan["City"].astype(str).str.strip().str.upper()

    penjualan["Tgl Faktur"] = pd.to_datetime(penjualan["Tgl Faktur"], errors="coerce")

    with st.expander("Lihat Data Penjualan Setelah Preprocessing"):
        preview_cols = ["No. Faktur", "Tgl Faktur", "Nama Pelanggan", "No. Barang", "Faktur + Barang", "Kuantitas"]
        preview_cols = [c for c in preview_cols if c in penjualan.columns]
        st.dataframe(penjualan[preview_cols].head(20), use_container_width=True)
        st.download_button(
            "📥 Unduh Data Penjualan Bersih (Excel)",
            data=convert_df_to_excel(penjualan),
            file_name="data_penjualan_bersih.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    st.markdown("---")

    # ── Tanggal Analisis ───────────────────────────────────────────────────────
    default_end = penjualan["Tgl Faktur"].dropna().max().date()
    if st.session_state.stock_filename:
        m = re.search(r"(\d{8})", st.session_state.stock_filename)
        if m:
            try:
                from datetime import datetime
                default_end = datetime.strptime(m.group(1), "%d%m%Y").date()
            except ValueError:
                pass
    default_start = default_end - timedelta(days=89)

    col1, col2 = st.columns(2)
    col1.date_input("Tanggal Awal (90 Hari Belakang)", value=default_start, key="stock_start", disabled=True)
    end_date = col2.date_input("Tanggal Akhir", value=default_end, key="stock_end")

    # ── Tombol Jalankan Analisa ────────────────────────────────────────────────
    if st.button("Jalankan Analisa Stock"):
        _run_stock_analysis(penjualan, produk_ref, df_stock, end_date)

    # ── Tampilkan Hasil ────────────────────────────────────────────────────────
    if st.session_state.stock_analysis_result is not None:
        _render_results()


# ── Logika Analisis Utama ──────────────────────────────────────────────────────
def _run_stock_analysis(penjualan, produk_ref, df_stock, end_date):
    with st.spinner("Melakukan perhitungan analisis stok..."):
        end_date_dt  = pd.to_datetime(end_date)
        wma_start    = end_date_dt - pd.DateOffset(days=89)
        penjualan_90 = penjualan[penjualan["Tgl Faktur"].between(wma_start, end_date_dt)]

        if penjualan_90.empty:
            st.error("Tidak ada data penjualan dalam rentang 90 hari terakhir.")
            st.session_state.stock_analysis_result = None
            return

        # Rentang bulanan
        r1_end   = end_date_dt
        r1_start = end_date_dt - pd.DateOffset(days=29)
        r2_end   = end_date_dt - pd.DateOffset(days=30)
        r2_start = end_date_dt - pd.DateOffset(days=59)
        r3_end   = end_date_dt - pd.DateOffset(days=60)
        r3_start = end_date_dt - pd.DateOffset(days=89)

        def _sales(start, end, col_name):
            return (
                penjualan_90[penjualan_90["Tgl Faktur"].between(start, end)]
                .groupby(["City", "No. Barang"])["Kuantitas"]
                .sum()
                .reset_index(name=col_name)
            )

        sales_m1 = _sales(r1_start, r1_end, "Penjualan Bln 1")
        sales_m2 = _sales(r2_start, r2_end, "Penjualan Bln 2")
        sales_m3 = _sales(r3_start, r3_end, "Penjualan Bln 3")

        total_90 = (
            penjualan_90.groupby(["City", "No. Barang"])["Kuantitas"]
            .sum()
            .reset_index()
        )
        total_90["AVG Mean"] = total_90["Kuantitas"] / 3
        total_90.drop("Kuantitas", axis=1, inplace=True)

        wma_grouped = (
            penjualan_90.groupby(["City", "No. Barang"])
            .apply(calculate_daily_wma, end_date=end_date)
            .reset_index(name="AVG WMA")
        )

        # Kombinasi lengkap City × Barang
        barang_list = produk_ref[["No. Barang", "Kategori Barang", "BRAND Barang", "Nama Barang"]].drop_duplicates()
        city_list   = penjualan["City"].unique()
        kombinasi   = pd.MultiIndex.from_product(
            [city_list, barang_list["No. Barang"]], names=["City", "No. Barang"]
        ).to_frame(index=False)
        full_data = pd.merge(kombinasi, barang_list, on="No. Barang", how="left")

        for df_merge in [wma_grouped, sales_m1, sales_m2, sales_m3, total_90]:
            full_data = pd.merge(full_data, df_merge, on=["City", "No. Barang"], how="left")

        # Kolom bulanan
        penjualan_90 = penjualan_90.copy()
        penjualan_90["Bulan"] = penjualan_90["Tgl Faktur"].dt.to_period("M")
        monthly = (
            penjualan_90.groupby(["City", "No. Barang", "Bulan"])["Kuantitas"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        full_data = pd.merge(full_data, monthly, on=["City", "No. Barang"], how="left")
        for col in full_data.columns:
            if pd.api.types.is_numeric_dtype(full_data[col]):
                full_data[col] = full_data[col].fillna(0)
            else:
                full_data[col] = full_data[col].fillna("")

        # Rename kolom Period → nama bulan Indonesia
        period_cols = sorted([c for c in full_data.columns if isinstance(c, pd.Period)])
        rename_map  = {c: f"{BULAN_INDONESIA[c.month]} {c.year}" for c in period_cols}
        full_data.rename(columns=rename_map, inplace=True)
        bulan_columns_renamed = [rename_map[c] for c in period_cols]

        full_data.rename(columns={"AVG WMA": "SO WMA", "AVG Mean": "SO Mean"}, inplace=True)
        full_data["SO Total"] = full_data["SO WMA"]

        # ── ABC Log-Benchmark ──────────────────────────────────────────────────
        log_df = classify_abc_log_benchmark(full_data.copy(), metric_col="SO WMA")
        log_cols_to_merge = [
            "City", "No. Barang",
            "Kategori ABC (Log-Benchmark - WMA)", "Ratio Log WMA", "Log (10) WMA", "Avg Log WMA",
        ]
        full_data = pd.merge(full_data, log_df[log_cols_to_merge], on=["City", "No. Barang"], how="left")

        # ── Min & Max Stock (Vectorized — FIX performa) ────────────────────────
        KAT_COL = "Kategori ABC (Log-Benchmark - WMA)"
        full_data["Min Stock"] = calculate_min_stock(full_data, KAT_COL, "SO WMA")
        full_data["Max Stock"] = calculate_max_stock(full_data, KAT_COL, "SO WMA")

        st.markdown("### ⚙️ Konfigurasi Stok Minimal")
        st.info("""
        ✅ **Metode Terkunci: Min Stock (Days / Time Based)**

        Perhitungan otomatis menggunakan multiplier waktu (WMA × Hari):
        * **A & B:** 1.00x (Buffer 30 Hari)
        * **C:** 0.75x (Buffer 24 Hari)
        * **D:** 0.50x (Buffer 15 Hari)
        * **E:** 0.25x (Buffer 7 Hari)
        * **F:** 0.0x  (Min Stock 0, Max Stock 1)
        """)

        # ── Stock Cabang ───────────────────────────────────────────────────────
        stock_df_raw  = df_stock.rename(columns=lambda x: x.strip())
        stock_melted  = melt_stock_by_city(stock_df_raw)

        full_data = pd.merge(
            full_data, stock_melted, on=["City", "No. Barang"], how="left"
        ).rename(columns={"Stock": "Stock Cabang"})
        full_data["Stock Cabang"] = full_data["Stock Cabang"].fillna(0)

        full_data["Status Stock"] = full_data.apply(get_status_stock, axis=1)

        # ── Add Stock (Vectorized) ─────────────────────────────────────────────
        full_data["Add Stock"] = calculate_add_stock(full_data, KAT_COL, "Min Stock", "Stock Cabang")

        # ── Suggested PO per Cabang (Proporsional) ────────────────────────────
        # Merge Stock Surabaya sebagai sumber distribusi
        stock_sby = stock_melted[stock_melted["City"] == "SURABAYA"][["No. Barang", "Stock"]].rename(columns={"Stock": "Stock Surabaya"})
        full_data = full_data.merge(stock_sby, on="No. Barang", how="left")
        full_data["Stock Surabaya"] = full_data["Stock Surabaya"].fillna(0)

        # Hitung Suggested PO proporsional (vectorized per No. Barang)
        full_data["Suggested PO"] = calculate_suggested_po(full_data)

        # ── Pembulatan ─────────────────────────────────────────────────────────
        int_cols = [
            "Stock Cabang", "Min Stock", "Max Stock", "Add Stock",
            "Suggested PO", "Stock Surabaya",
            "SO WMA", "SO Mean", "Penjualan Bln 1", "Penjualan Bln 2", "Penjualan Bln 3",
        ] + bulan_columns_renamed
        for col in int_cols:
            if col in full_data.columns:
                full_data[col] = full_data[col].round(0).astype(int)

        for col in ["Log (10) WMA", "Avg Log WMA", "Ratio Log WMA"]:
            if col in full_data.columns:
                full_data[col] = full_data[col].round(2)

        st.session_state.stock_analysis_result = full_data.copy()
        st.session_state.bulan_columns_stock   = bulan_columns_renamed
        st.success("✅ Analisis Stok berhasil dijalankan!")


# ── Render Hasil Tabel & Dashboard ────────────────────────────────────────────
def _render_results():
    result     = st.session_state.stock_analysis_result.copy()
    result     = result[result["City"] != "OTHERS"]
    bulan_cols = st.session_state.get("bulan_columns_stock", [])

    st.markdown("---")
    st.header("Filter Produk (Berlaku untuk Semua Tabel)")
    col_f1, col_f2, col_f3 = st.columns(3)
    sel_kat  = col_f1.multiselect("Kategori:",    sorted(result["Kategori Barang"].dropna().unique().astype(str)))
    sel_brand = col_f2.multiselect("Brand:",       sorted(result["BRAND Barang"].dropna().unique().astype(str)))
    sel_prod  = col_f3.multiselect("Nama Produk:", sorted(result["Nama Barang"].dropna().unique().astype(str)))

    if sel_kat:   result = result[result["Kategori Barang"].astype(str).isin(sel_kat)]
    if sel_brand: result = result[result["BRAND Barang"].astype(str).isin(sel_brand)]
    if sel_prod:  result = result[result["Nama Barang"].astype(str).isin(sel_prod)]

    st.header("Filter Hasil (Hanya untuk Tabel per Kota)")
    col_h1, col_h2 = st.columns(2)
    sel_abc    = col_h1.multiselect("Kategori ABC:", sorted(result["Kategori ABC (Log-Benchmark - WMA)"].dropna().unique().astype(str)))
    sel_status = col_h2.multiselect("Status Stock:", sorted(result["Status Stock"].dropna().unique().astype(str)))

    st.markdown("---")
    tab1, tab2 = st.tabs(["Hasil Tabel", "Dashboard"])

    with tab1:
        _render_table(result, bulan_cols, sel_abc, sel_status)
    with tab2:
        _render_dashboard(result)


def _render_table(result, bulan_cols, sel_abc, sel_status):
    header_style = {"selector": "th", "props": [("background-color", "#0068c9"), ("color", "white"), ("text-align", "center")]}
    st.header("Hasil Analisis Stok per Kota")
    KEYS = ["No. Barang", "Kategori Barang", "BRAND Barang", "Nama Barang"]

    for city in sorted(result["City"].unique()):
        with st.expander(f"📍 Lihat Hasil Stok untuk Kota: {city}"):
            city_df = result[result["City"] == city].copy()
            if sel_abc:    city_df = city_df[city_df["Kategori ABC (Log-Benchmark - WMA)"].isin(sel_abc)]
            if sel_status: city_df = city_df[city_df["Status Stock"].isin(sel_status)]
            if city_df.empty:
                st.write("Tidak ada data yang cocok dengan filter yang dipilih.")
                continue

            metric_order = (
                bulan_cols
                + ["Penjualan Bln 1", "Penjualan Bln 2", "Penjualan Bln 3"]
                + ["SO WMA", "SO Mean", "SO Total"]
                + ["Log (10) WMA", "Avg Log WMA", "Ratio Log WMA", "Kategori ABC (Log-Benchmark - WMA)"]
                + ["Min Stock", "Max Stock", "Stock Cabang", "Status Stock", "Add Stock", "Suggested PO"]
            )
            display_cols = KEYS + [c for c in metric_order if c in city_df.columns]
            city_df = city_df[display_cols]

            fmt = {}
            skip = set(KEYS)
            for col in city_df.columns:
                if col in skip or not pd.api.types.is_numeric_dtype(city_df[col]):
                    continue
                fmt[col] = "{:.2f}" if any(x in col for x in ["Ratio", "Log", "Avg Log"]) else "{:.0f}"

            st.dataframe(
                city_df.style
                .format(fmt, na_rep="-")
                .apply(lambda x: x.map(highlight_kategori_abc_log), subset=["Kategori ABC (Log-Benchmark - WMA)"])
                .apply(lambda x: x.map(highlight_status_stock),     subset=["Status Stock"])
                .set_table_styles([header_style]),
                use_container_width=True,
            )

    # Tabel Gabungan Pivot
    st.header("📊 Tabel Gabungan Seluruh Kota (Stock)")
    with st.spinner("Membuat tabel pivot gabungan..."):
        if result.empty:
            st.warning("Tidak ada data untuk tabel gabungan.")
            return

        _render_pivot_table(result, bulan_cols, KEYS)

    # Unduh
    st.header("💾 Unduh Hasil Analisis Stock")
    output_stock = BytesIO()
    with pd.ExcelWriter(output_stock, engine="openpyxl") as writer:
        # Sheet 1: Tabel pivot gabungan semua kota (df_style dari _render_pivot_table)
        # Karena df_style ada di dalam _render_pivot_table, kita simpan ke session state
        if "stock_pivot_df" in st.session_state and not st.session_state.stock_pivot_df.empty:
            st.session_state.stock_pivot_df.to_excel(writer, sheet_name="All Cities Pivot", index=False)
        # Sheet 2: Data filtered lengkap
        result.to_excel(writer, sheet_name="Filtered Data", index=False)
        # Sheet per kota
        for city in sorted(result["City"].unique()):
            city_df = result[result["City"] == city]
            if not city_df.empty:
                city_df.to_excel(writer, sheet_name=city[:31], index=False)
    st.download_button(
        "📥 Unduh Hasil Analisis Stock (Excel)",
        data=output_stock.getvalue(),
        file_name=f"Hasil_Analisis_Stock_{st.session_state.get('stock_end', 'export')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _render_pivot_table(result, bulan_cols, KEYS):
    from utils import classify_abc_log_benchmark, get_days_multiplier

    pivot_cols = (
        bulan_cols
        + ["Penjualan Bln 1", "Penjualan Bln 2", "Penjualan Bln 3"]
        + ["SO WMA", "SO Mean", "SO Total"]
        + ["Log (10) WMA", "Avg Log WMA", "Ratio Log WMA", "Kategori ABC (Log-Benchmark - WMA)"]
        + ["Min Stock", "Max Stock", "Stock Cabang", "Status Stock", "Add Stock", "Suggested PO"]
    )
    pivot_cols_existing = [col for col in pivot_cols if col in result.columns]
    pivot_result = result.pivot_table(index=KEYS, columns="City", values=pivot_cols_existing, aggfunc="first")
    pivot_result.columns = [f"{level1}_{level0}" for level0, level1 in pivot_result.columns]
    pivot_result.reset_index(inplace=True)

    cities = sorted(result["City"].unique())
    metric_order = pivot_cols
    ordered_city_cols = [f"{city}_{metric}" for city in cities for metric in metric_order]
    existing_ordered_cols = [col for col in ordered_city_cols if col in pivot_result.columns]

    # ALL summary
    total_agg = result.groupby(KEYS).agg(
        All_Stock=("Stock Cabang", "sum"),
        All_SO=("SO WMA", "sum"),
    ).reset_index()

    # Pastikan tidak ada spasi aneh
    result.columns = result.columns.str.strip()

    # Hitung DNA ABC lengkap untuk kategori ALL
    all_sales_for_abc = result.groupby(KEYS, as_index=False).agg({"SO WMA": "sum"})
    all_sales_for_abc.rename(columns={"SO WMA": "Total Kuantitas"}, inplace=True)
    all_sales_for_abc["City"] = "ALL"

    all_classified = classify_abc_log_benchmark(all_sales_for_abc, metric_col="Total Kuantitas")
    all_classified.rename(columns={
        "Log (10) Total Kuantitas":                       "All_Log",
        "Avg Log Total Kuantitas":                        "All_Avg Log",
        "Ratio Log Total Kuantitas":                      "All_Ratio",
        "Kategori ABC (Log-Benchmark - Total Kuantitas)": "All_Kategori ABC All",
    }, inplace=True)

    # Merge kategori ke total_agg lalu hitung All_Add_Stock
    total_agg = pd.merge(total_agg, all_classified[KEYS + ["All_Kategori ABC All"]], on=KEYS, how="left")

    def calc_all_add_stock(row):
        if row["All_Kategori ABC All"] == "F":
            return 0
        mult = get_days_multiplier(row["All_Kategori ABC All"])
        return max(0, math.ceil(row["All_SO"] * mult) - row["All_Stock"])

    total_agg["All_Add_Stock"]       = total_agg.apply(calc_all_add_stock, axis=1)
    total_agg["All_Restock 1 Bulan"] = np.where(total_agg["All_Stock"] < total_agg["All_SO"], "PO", "NO")

    pivot_result = pd.merge(pivot_result, total_agg, on=KEYS, how="left")
    pivot_result = pd.merge(pivot_result, all_classified[KEYS + ["All_Log", "All_Avg Log", "All_Ratio"]], on=KEYS, how="left")

    # Daftarkan kolom ALL di summary
    final_summary_cols = [
        "All_Stock", "All_SO", "All_Add_Stock",
        "All_Log", "All_Avg Log", "All_Ratio", "All_Kategori ABC All", "All_Restock 1 Bulan"
    ]
    final_display_cols = KEYS + existing_ordered_cols + final_summary_cols
    df_style = pivot_result[[c for c in final_display_cols if c in pivot_result.columns]].copy()

    numeric_cols_to_format = []
    float_cols_to_format   = []
    object_cols_to_format  = []
    for col in df_style.columns:
        if col not in KEYS:
            if any(x in col for x in ["Ratio", "Log", "Avg Log"]):
                float_cols_to_format.append(col)
            elif pd.api.types.is_numeric_dtype(df_style[col]):
                numeric_cols_to_format.append(col)
            else:
                object_cols_to_format.append(col)

    df_style[numeric_cols_to_format] = df_style[numeric_cols_to_format].fillna(0).astype(int)
    df_style[float_cols_to_format]   = df_style[float_cols_to_format].fillna(0)
    df_style[object_cols_to_format]  = df_style[object_cols_to_format].fillna("-")

    # Simpan ke session state agar bisa dipakai di tombol download
    st.session_state.stock_pivot_df = df_style.copy()

    column_config_stock = {}
    for col in numeric_cols_to_format: column_config_stock[col] = st.column_config.NumberColumn(format="%.0f")
    for col in float_cols_to_format:   column_config_stock[col] = st.column_config.NumberColumn(format="%.2f")

    st.dataframe(df_style, column_config=column_config_stock, use_container_width=True)


def _render_dashboard(result):
    st.header("📈 Dashboard Analisis Stock")
    if result.empty:
        st.info("Tidak ada data untuk ditampilkan.")
        return

    total_under = result[result["Status Stock"] == "Understock"].shape[0]
    total_over  = result[result["Status Stock"].str.contains("Overstock", na=False)].shape[0]
    col1, col2  = st.columns(2)
    col1.metric("Total Produk Understock", f"{total_under} SKU")
    col2.metric("Total Produk Overstock",  f"{total_over} SKU")
    st.markdown("---")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("Distribusi Kategori ABC (Log-Benchmark)")
        st.bar_chart(result["Kategori ABC (Log-Benchmark - WMA)"].value_counts())
    with col_c2:
        st.subheader("Distribusi Status Stok")
        st.bar_chart(result["Status Stock"].value_counts())

    st.markdown("---")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("Top 5 Produk Paling Understock")
        top_u = (
            result[result["Status Stock"] == "Understock"]
            .sort_values("Add Stock", ascending=False)
            .head(5)
        )
        st.dataframe(top_u[["Nama Barang", "City", "Add Stock", "Stock Cabang", "Min Stock"]], use_container_width=True)
    with col_t2:
        st.subheader("Top 5 Produk Paling Overstock")
        ov = result[result["Status Stock"].str.contains("Overstock", na=False)].copy()
        ov["Kelebihan Stok"] = ov["Stock Cabang"] - ov["Max Stock"]
        st.dataframe(
            ov.sort_values("Kelebihan Stok", ascending=False).head(5)
            [["Nama Barang", "City", "Kelebihan Stok", "Stock Cabang", "Max Stock"]],
            use_container_width=True,
        )
