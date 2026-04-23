"""
pages/abc_analysis.py
Halaman Hasil Analisa ABC — dua metode Log-Benchmark (Mean & WMA).
"""

import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt

from utils import (
    map_nama_dept,
    map_city,
    classify_abc_log_benchmark,
    highlight_kategori_abc_log,
)


def render():
    st.title("📊 Analisis ABC Berdasarkan Metrik Penjualan Dinamis (Log-Benchmark)")
    tab1, tab2 = st.tabs(["Hasil Tabel", "Dashboard"])

    with tab1:
        _render_table_tab()
    with tab2:
        _render_dashboard_tab()


# ── Tab Tabel ──────────────────────────────────────────────────────────────────
def _render_table_tab():
    if st.session_state.df_penjualan.empty or st.session_state.produk_ref.empty:
        st.warning("⚠️ Harap muat file **Penjualan** dan **Produk Referensi** di halaman **'Input Data'** terlebih dahulu.")
        st.stop()

    # Preprocessing
    so_df      = st.session_state.df_penjualan.copy()
    produk_ref = st.session_state.produk_ref.copy()

    for df in [so_df, produk_ref]:
        if "No. Barang" in df.columns:
            df["No. Barang"] = df["No. Barang"].astype(str).str.strip()

    so_df.rename(columns={"Qty": "Kuantitas"}, inplace=True, errors="ignore")
    so_df["Nama Dept"] = so_df.apply(map_nama_dept, axis=1)
    so_df["City"]      = so_df["Nama Dept"].apply(map_city)

    if "Kategori Barang" in produk_ref.columns:
        produk_ref["Kategori Barang"] = produk_ref["Kategori Barang"].astype(str).str.strip().str.upper()
    if "City" in so_df.columns:
        so_df["City"] = so_df["City"].astype(str).str.strip().str.upper()

    so_df["Tgl Faktur"] = pd.to_datetime(so_df["Tgl Faktur"], dayfirst=True, errors="coerce")
    so_df.dropna(subset=["Tgl Faktur"], inplace=True)

    # Filter tanggal
    st.header("Filter Rentang Waktu Analisis ABC")
    st.info("Analisis akan didasarkan pada data penjualan 90 hari *sebelum* **Tanggal Akhir** yang dipilih.")
    min_date = so_df["Tgl Faktur"].min().date()
    max_date = so_df["Tgl Faktur"].max().date()
    end_date_input = st.date_input("Tanggal Akhir", value=max_date, min_value=min_date, max_value=max_date)

    if st.button("Jalankan Analisa ABC (2 Metode Log-Benchmark)"):
        _run_abc_analysis(so_df, produk_ref, end_date_input)

    if st.session_state.abc_analysis_result is None:
        return

    result_display = st.session_state.abc_analysis_result.copy()
    result_display = result_display[result_display["City"] != "OTHERS"]

    # Filter
    st.header("Filter Hasil Analisis")
    col_f1, col_f2 = st.columns(2)
    sel_kat   = col_f1.multiselect("Filter Kategori:",  sorted(produk_ref["Kategori Barang"].dropna().unique().astype(str)), key="abc_cat_filter")
    sel_brand = col_f2.multiselect("Filter Brand:",     sorted(produk_ref["BRAND Barang"].dropna().unique().astype(str)),    key="abc_brand_filter")
    if sel_kat:   result_display = result_display[result_display["Kategori Barang"].astype(str).isin(sel_kat)]
    if sel_brand: result_display = result_display[result_display["BRAND Barang"].astype(str).isin(sel_brand)]

    KEYS = ["No. Barang", "Kategori Barang", "BRAND Barang", "Nama Barang"]

    # Tabel per kota
    st.header("Hasil Analisis ABC per Kota")
    for city in sorted(result_display["City"].unique()):
        with st.expander(f"🏙️ Lihat Hasil ABC untuk Kota: {city}"):
            city_df = result_display[result_display["City"] == city]
            col_order = [
                "No. Barang", "BRAND Barang", "Nama Barang", "Kategori Barang",
                "AVG Mean", "AVG WMA",
                "Kategori ABC (Log-Benchmark - Mean)", "Kategori ABC (Log-Benchmark - WMA)",
                "Log (10) Mean", "Avg Log Mean", "Ratio Log Mean",
                "Log (10) WMA",  "Avg Log WMA",  "Ratio Log WMA",
            ]
            display_cols = [c for c in col_order if c in city_df.columns]
            df_show = city_df[display_cols]

            fmt = {}
            for col in df_show.columns:
                if col in KEYS or not pd.api.types.is_numeric_dtype(df_show[col]):
                    continue
                fmt[col] = "{:.2f}" if any(x in col for x in ["Ratio", "Log", "Avg Log"]) else "{:.0f}"

            style = df_show.style.format(fmt, na_rep="-")
            if "Kategori ABC (Log-Benchmark - Mean)" in df_show.columns:
                style = style.apply(lambda x: x.map(highlight_kategori_abc_log), subset=["Kategori ABC (Log-Benchmark - Mean)"])
            if "Kategori ABC (Log-Benchmark - WMA)" in df_show.columns:
                style = style.apply(lambda x: x.map(highlight_kategori_abc_log), subset=["Kategori ABC (Log-Benchmark - WMA)"])
            st.dataframe(style, use_container_width=True)

    # Tabel Pivot Gabungan
    st.header("📊 Tabel Gabungan Seluruh Kota (ABC)")
    with st.spinner("Membuat tabel pivot gabungan..."):
        _render_pivot_abc(result_display, KEYS, end_date_input)


def _run_abc_analysis(so_df, produk_ref, end_date_input):
    with st.spinner("Melakukan perhitungan analisis ABC..."):
        end_dt   = pd.to_datetime(end_date_input)
        start_90 = end_dt - pd.DateOffset(days=89)
        df_90    = so_df[so_df["Tgl Faktur"].between(start_90, end_dt)]

        if df_90.empty:
            st.error("Tidak ada data penjualan pada rentang 90 hari yang dipilih.")
            st.session_state.abc_analysis_result = None
            return

        def _sales(start, end, col):
            return (
                df_90[df_90["Tgl Faktur"].between(start, end)]
                .groupby(["City", "No. Barang"])["Kuantitas"]
                .sum().reset_index(name=col)
            )

        r1_end, r1_start = end_dt, end_dt - pd.DateOffset(days=29)
        r2_end, r2_start = end_dt - pd.DateOffset(days=30), end_dt - pd.DateOffset(days=59)
        r3_end, r3_start = end_dt - pd.DateOffset(days=60), end_dt - pd.DateOffset(days=89)

        sales_m1 = _sales(r1_start, r1_end, "Penjualan Bln 1")
        sales_m2 = _sales(r2_start, r2_end, "Penjualan Bln 2")
        sales_m3 = _sales(r3_start, r3_end, "Penjualan Bln 3")

        produk_ref.rename(columns={"Keterangan Barang": "Nama Barang", "Nama Kategori Barang": "Kategori Barang"}, inplace=True, errors="ignore")
        barang_list = produk_ref[["No. Barang", "BRAND Barang", "Kategori Barang", "Nama Barang"]].drop_duplicates()
        city_list   = so_df["City"].dropna().unique()

        kombinasi = pd.MultiIndex.from_product([city_list, barang_list["No. Barang"]], names=["City", "No. Barang"]).to_frame(index=False)
        grouped   = pd.merge(kombinasi, barang_list, on="No. Barang", how="left")
        for sm in [sales_m1, sales_m2, sales_m3]:
            grouped = pd.merge(grouped, sm, on=["City", "No. Barang"], how="left")
        grouped.fillna({"Penjualan Bln 1": 0, "Penjualan Bln 2": 0, "Penjualan Bln 3": 0}, inplace=True)

        grouped["AVG Mean"] = (grouped["Penjualan Bln 1"] + grouped["Penjualan Bln 2"] + grouped["Penjualan Bln 3"]) / 3
        grouped["AVG WMA"]  = np.ceil(
            grouped["Penjualan Bln 1"] * 0.5
            + grouped["Penjualan Bln 2"] * 0.3
            + grouped["Penjualan Bln 3"] * 0.2
        )

        res_mean = classify_abc_log_benchmark(grouped.copy(), metric_col="AVG Mean")
        res_wma  = classify_abc_log_benchmark(grouped.copy(), metric_col="AVG WMA")

        MERGE_KEYS = ["City", "No. Barang", "BRAND Barang", "Kategori Barang", "Nama Barang",
                      "Penjualan Bln 1", "Penjualan Bln 2", "Penjualan Bln 3", "AVG Mean", "AVG WMA"]
        result_final = res_mean.copy()
        wma_extra = [c for c in res_wma.columns
                     if any(x in c for x in ["Log-Benchmark - WMA", "Log (10) WMA", "Avg Log WMA", "Ratio Log WMA"])
                     and c not in result_final.columns]
        result_final = pd.merge(result_final, res_wma[["City", "No. Barang"] + wma_extra], on=["City", "No. Barang"], how="left")

        for col in ["Penjualan Bln 1", "Penjualan Bln 2", "Penjualan Bln 3", "AVG Mean", "AVG WMA"]:
            if col in result_final.columns:
                result_final[col] = result_final[col].round(0).astype(int)
        for col in ["Log (10) WMA", "Avg Log WMA", "Ratio Log WMA", "Log (10) Mean", "Avg Log Mean", "Ratio Log Mean"]:
            if col in result_final.columns:
                result_final[col] = result_final[col].round(2)

        st.session_state.abc_analysis_result = result_final.copy()
        st.success("✅ Analisis ABC (2 Metode Log-Benchmark) berhasil dijalankan!")


def _render_pivot_abc(result_display, KEYS, end_date_input):
    pivot_values = [
        "Penjualan Bln 1", "Penjualan Bln 2", "Penjualan Bln 3", "AVG Mean", "AVG WMA",
        "Kategori ABC (Log-Benchmark - Mean)", "Ratio Log Mean", "Log (10) Mean", "Avg Log Mean",
        "Kategori ABC (Log-Benchmark - WMA)",  "Ratio Log WMA",  "Log (10) WMA",  "Avg Log WMA",
    ]
    existing_vals = [c for c in pivot_values if c in result_display.columns]
    pivot = result_display.pivot_table(index=KEYS, columns="City", values=existing_vals, aggfunc="first")
    pivot.columns = [f"{lv1}_{lv0}" for lv0, lv1 in pivot.columns]
    pivot.reset_index(inplace=True)

    # ALL summary
    total = result_display.groupby(KEYS).agg({
        "Penjualan Bln 1": "sum", "Penjualan Bln 2": "sum", "Penjualan Bln 3": "sum"
    }).reset_index()
    total["AVG Mean"] = (total["Penjualan Bln 1"] + total["Penjualan Bln 2"] + total["Penjualan Bln 3"]) / 3
    total["AVG WMA"]  = np.ceil(total["Penjualan Bln 1"] * 0.5 + total["Penjualan Bln 2"] * 0.3 + total["Penjualan Bln 3"] * 0.2)

    for col in ["Penjualan Bln 1", "Penjualan Bln 2", "Penjualan Bln 3", "AVG Mean", "AVG WMA"]:
        if col in total.columns:
            total[col] = total[col].round(0).astype(int)

    total["City"] = "ALL"
    abc_mean_all = classify_abc_log_benchmark(total.copy(), metric_col="AVG Mean")
    abc_wma_all  = classify_abc_log_benchmark(total.copy(), metric_col="AVG WMA")

    total_final = total.drop(columns=["City"], errors="ignore")
    for src, kw in [(abc_mean_all, ["Log-Benchmark - Mean", "Log (10) Mean", "Avg Log Mean", "Ratio Log Mean"]),
                    (abc_wma_all,  ["Log-Benchmark - WMA",  "Log (10) WMA",  "Avg Log WMA",  "Ratio Log WMA"])]:
        extra_cols = [c for c in src.columns if any(x in c for x in kw)]
        total_final = pd.merge(total_final, src[KEYS + extra_cols], on=KEYS, how="left")

    for col in ["Log (10) WMA", "Avg Log WMA", "Ratio Log WMA", "Log (10) Mean", "Avg Log Mean", "Ratio Log Mean"]:
        if col in total_final.columns:
            total_final[col] = total_final[col].round(2)

    total_final.columns = [f"All_{c}" if c not in KEYS else c for c in total_final.columns]
    pivot_final = pd.merge(pivot, total_final, on=KEYS, how="left")

    df_style = pivot_final.copy()
    num_cols   = [c for c in df_style.columns if c not in KEYS and pd.api.types.is_numeric_dtype(df_style[c])
                  and not any(x in c for x in ["Ratio", "Log", "Avg Log"])]
    float_cols = [c for c in df_style.columns if c not in KEYS and any(x in c for x in ["Ratio", "Log", "Avg Log"])]
    obj_cols   = [c for c in df_style.columns if c not in KEYS and c not in num_cols and c not in float_cols]

    df_style[num_cols]   = df_style[num_cols].fillna(0).astype(int)
    df_style[float_cols] = df_style[float_cols].fillna(0)
    df_style[obj_cols]   = df_style[obj_cols].fillna("-")

    col_cfg = {}
    for c in num_cols:   col_cfg[c] = st.column_config.NumberColumn(format="%.0f")
    for c in float_cols: col_cfg[c] = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(df_style, column_config=col_cfg, use_container_width=True)

    # Download
    st.header("💾 Unduh Hasil Analisis ABC")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_style.to_excel(writer, sheet_name="All Cities Pivot", index=False)
        for city in sorted(result_display["City"].unique()):
            result_display[result_display["City"] == city].to_excel(writer, sheet_name=city[:31], index=False)
    st.download_button(
        "📥 Unduh Hasil Analisis ABC (Excel)",
        data=output.getvalue(),
        file_name=f"Hasil_Analisis_ABC_{end_date_input}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ── Tab Dashboard ──────────────────────────────────────────────────────────────
def _render_dashboard_tab():
    if st.session_state.abc_analysis_result is None:
        st.info("Tidak ada data untuk ditampilkan. Jalankan analisis terlebih dahulu.")
        return

    result = st.session_state.abc_analysis_result.copy()
    metode = st.selectbox("Pilih Metode ABC untuk Dashboard:", ("Log-Benchmark - WMA", "Log-Benchmark - Mean"))

    if metode == "Log-Benchmark - WMA":
        kat_col, metric_col = "Kategori ABC (Log-Benchmark - WMA)", "AVG WMA"
    else:
        kat_col, metric_col = "Kategori ABC (Log-Benchmark - Mean)", "AVG Mean"

    if result.empty:
        st.info("Tidak ada data untuk ditampilkan.")
        return

    LABELS = ["A", "B", "C", "D", "E", "F"]
    COLORS = ["#cce5ff", "#d4edda", "#fff3cd", "#f8d7da", "#e9ecef", "#6c757d"]

    summary = result.groupby(kat_col)[metric_col].agg(["count", "sum"])
    for label in LABELS:
        if label not in summary.index:
            summary.loc[label] = [0, 0]
    summary = summary.reindex(LABELS).fillna(0)
    total_sum = summary["sum"].sum()
    summary["avg_unit"] = np.where(summary["count"] > 0, summary["sum"] / summary["count"], 0)

    # Metric cards
    st.markdown("---")
    cols = st.columns(len(LABELS))
    for i, label in enumerate(LABELS):
        count = int(summary.loc[label, "count"])
        avg   = summary.loc[label, "avg_unit"]
        delta = "Tidak Terjual" if label == "F" else f"{avg:.1f} Rata-rata Penjualan"
        cols[i].metric(f"Produk Kelas {label}", f"{count} SKU", delta)

    st.markdown("---")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("Komposisi Produk per Kelas (SKU Count)")
        data_pie = summary[summary["count"] > 0]
        if not data_pie.empty:
            fig, ax = plt.subplots()
            ax.pie(data_pie["count"], labels=data_pie.index, autopct="%1.1f%%", startangle=90,
                   colors=[COLORS[LABELS.index(i)] for i in data_pie.index])
            ax.axis("equal")
            st.pyplot(fig)
        else:
            st.info("Tidak ada data untuk pie chart.")

    with col_c2:
        st.subheader(f"Kontribusi {metric_col} per Kelas")
        data_bar = summary[summary["sum"] > 0]
        if not data_bar.empty:
            st.bar_chart(data_bar[["sum"]].rename(columns={"sum": metric_col}))
        else:
            st.info("Tidak ada kontribusi penjualan untuk ditampilkan.")

    st.markdown("---")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader(f"Top 10 Produk Terlaris (berdasarkan {metric_col})")
        top = result.groupby("Nama Barang")[metric_col].sum().nlargest(10)
        st.bar_chart(top)
    with col_t2:
        st.subheader(f"Performa Penjualan per Kota (berdasarkan {metric_col})")
        city_sales = result.groupby("City")[metric_col].sum().sort_values(ascending=False)
        st.bar_chart(city_sales)
