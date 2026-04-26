"""
pages/stock_analysis_v2.py
Halaman Hasil Analisa Stock V2 — Logika baru dengan:
  - Buffer lead time 7 hari di Add Stock
  - 4 skenario distribusi Suggested PO
  - All Add Stock terpisah (Cabang / Surabaya / Need Supplier)
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
    get_status_stock,
    melt_stock_by_city,
    highlight_kategori_abc_log,
    highlight_status_stock,
)
from utils.analysis import (
    calculate_add_stock_v2,
    calculate_suggested_po_v2,
    calculate_all_summary_v2,
    LEAD_TIME_CABANG,
    LEAD_TIME_SUPPLIER,
)


# ── Render Utama ───────────────────────────────────────────────────────────────
def render():
    st.title("📈 Hasil Analisa Stock V2")
    st.info(f"""
    **Perbedaan dengan V1:**
    - ✅ Suggested PO menggunakan **3 skenario distribusi** berdasarkan kondisi stok Surabaya
    - ✅ Distribusi proporsional berdasarkan **Kategori ABC → SO WMA**
    - ✅ Filter cabang **AMAN / TIDAK AMAN** (50% threshold)
    - ✅ All Add Stock dipisah: **Cabang / Surabaya / Need Supplier**
    - ✅ Surabaya tidak pernah kirim ke dirinya sendiri
    - ✅ Stok Sisa = Max(0, Stock Sby - Min Stock Sby)
    """)

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

    # ── Normalisasi Kolom Kunci ─────────────────────────────────────────────
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

    # ── Rename & Mapping ───────────────────────────────────────────────────
    penjualan.rename(columns={"Qty": "Kuantitas"}, inplace=True, errors="ignore")

    penjualan["Nama Dept"] = penjualan.apply(map_nama_dept, axis=1)
    penjualan["City"]      = penjualan["Nama Dept"].apply(map_city)

    produk_ref.rename(columns={"Keterangan Barang": "Nama Barang"}, inplace=True, errors="ignore")
    if "Kategori Barang" in produk_ref.columns:
        produk_ref["Kategori Barang"] = produk_ref["Kategori Barang"].astype(str).str.strip().str.upper()
    penjualan["City"]       = penjualan["City"].astype(str).str.strip().str.upper()
    penjualan["Tgl Faktur"] = pd.to_datetime(penjualan["Tgl Faktur"], errors="coerce")

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
    col1.date_input("Tanggal Awal (90 Hari Belakang)", value=default_start,
                    key="stock_v2_start", disabled=True)
    end_date = col2.date_input("Tanggal Akhir", value=default_end, key="stock_v2_end")

    if st.button("🚀 Jalankan Analisa Stock V2"):
        _run_analysis_v2(penjualan, produk_ref, df_stock, end_date)

    if st.session_state.get("stock_v2_result") is not None:
        _render_results_v2()


# ── Kalkulasi Utama ────────────────────────────────────────────────────────────
def _run_analysis_v2(penjualan, produk_ref, df_stock, end_date):
    with st.spinner("Melakukan perhitungan analisis stok V2..."):
        end_dt   = pd.to_datetime(end_date)
        wma_start = end_dt - pd.DateOffset(days=89)
        penjualan_90 = penjualan[penjualan["Tgl Faktur"].between(wma_start, end_dt)]

        if penjualan_90.empty:
            st.error("Tidak ada data penjualan dalam rentang 90 hari terakhir.")
            st.session_state["stock_v2_result"] = None
            return

        # Rentang bulanan
        r1_end, r1_start = end_dt, end_dt - pd.DateOffset(days=29)
        r2_end, r2_start = end_dt - pd.DateOffset(days=30), end_dt - pd.DateOffset(days=59)
        r3_end, r3_start = end_dt - pd.DateOffset(days=60), end_dt - pd.DateOffset(days=89)

        def _sales(start, end, col):
            df = (
                penjualan_90[penjualan_90["Tgl Faktur"].between(start, end)]
                .groupby(["City", "No. Barang"])["Kuantitas"]
                .sum().reset_index()
            )
            df.rename(columns={"Kuantitas": col}, inplace=True)
            return df

        sales_m1 = _sales(r1_start, r1_end, "Penjualan Bln 1")
        sales_m2 = _sales(r2_start, r2_end, "Penjualan Bln 2")
        sales_m3 = _sales(r3_start, r3_end, "Penjualan Bln 3")

        total_90 = (
            penjualan_90.groupby(["City", "No. Barang"])["Kuantitas"]
            .sum().reset_index()
        )
        total_90["AVG Mean"] = total_90["Kuantitas"] / 3
        total_90.drop("Kuantitas", axis=1, inplace=True)

        wma_grouped = (
            penjualan_90.groupby(["City", "No. Barang"])
            .apply(calculate_daily_wma, end_date=end_date)
            .reset_index()
        )
        wma_grouped.rename(columns={wma_grouped.columns[-1]: "AVG WMA"}, inplace=True)

        # Kombinasi lengkap City × Barang
        barang_list = produk_ref[["No. Barang", "Kategori Barang", "BRAND Barang", "Nama Barang"]].drop_duplicates()
        city_list   = penjualan["City"].unique()
        kombinasi   = pd.MultiIndex.from_product(
            [city_list, barang_list["No. Barang"]], names=["City", "No. Barang"]
        ).to_frame(index=False)
        full = pd.merge(kombinasi, barang_list, on="No. Barang", how="left")

        for dm in [wma_grouped, sales_m1, sales_m2, sales_m3, total_90]:
            full = pd.merge(full, dm, on=["City", "No. Barang"], how="left")

        # Kolom bulanan
        penjualan_90 = penjualan_90.copy()
        penjualan_90["Bulan"] = penjualan_90["Tgl Faktur"].dt.to_period("M")
        monthly = (
            penjualan_90.groupby(["City", "No. Barang", "Bulan"])["Kuantitas"]
            .sum().unstack(fill_value=0).reset_index()
        )
        full = pd.merge(full, monthly, on=["City", "No. Barang"], how="left")
        
        for col in full.columns:
            try:
                if pd.api.types.is_numeric_dtype(full[col]):
                    full[col] = full[col].fillna(0)
                else:
                    full[col] = full[col].astype(str).fillna("")
            except Exception as e:
                print(f"Error di kolom {col}: {e}")

        period_cols = sorted([c for c in full.columns if isinstance(c, pd.Period)])
        rename_map  = {c: f"{BULAN_INDONESIA[c.month]} {c.year}" for c in period_cols}
        full.rename(columns=rename_map, inplace=True)
        bulan_columns_renamed = [rename_map[c] for c in period_cols]

        full.rename(columns={"AVG WMA": "SO WMA", "AVG Mean": "SO Mean"}, inplace=True)
        full["SO Total"] = full["SO WMA"]

        # ABC Log-Benchmark
        KAT_COL = "Kategori ABC (Log-Benchmark - WMA)"
        log_df  = classify_abc_log_benchmark(full.copy(), metric_col="SO WMA")
        full    = pd.merge(
            full,
            log_df[["City", "No. Barang", KAT_COL, "Ratio Log WMA", "Log (10) WMA", "Avg Log WMA"]],
            on=["City", "No. Barang"], how="left"
        )

        # Min & Max Stock
        full["Min Stock"] = calculate_min_stock(full, KAT_COL, "SO WMA")
        full["Max Stock"] = calculate_max_stock(full, KAT_COL, "SO WMA")

        st.markdown("### ⚙️ Konfigurasi V2")
        st.info("""
        **Add Stock** = Max(0, Min Stock - Stock Cabang)

        **Stok Sisa Surabaya** = Max(0, Stock Sby - Min Stock Sby)

        **Skenario Distribusi:**
        - 🔴 **KURANG**: Stok Sisa = 0 → semua cabang PO = 0
        - 🟡 **SISA**: 0 < Stok Sisa < All Add Cabang → prioritas ABC → SO WMA
        - 🟢 **OVER**: Stok Sisa ≥ All Add Cabang → semua cabang dapat penuh

        **Multiplier Min Stock:**
        A & B = 1.00x | C = 0.75x | D = 0.50x | E = 0.25x | F = 0x
        """)

        # Stock Cabang
        stock_df_raw = df_stock.rename(columns=lambda x: x.strip())
        stock_melted = melt_stock_by_city(stock_df_raw)

        full = pd.merge(
            full, stock_melted, on=["City", "No. Barang"], how="left"
        ).rename(columns={"Stock": "Stock Cabang"})
        full["Stock Cabang"] = full["Stock Cabang"].fillna(0)

        full["Status Stock"] = full.apply(get_status_stock, axis=1)


        # ── All Stock Cabang & All SO Cabang (semua kota termasuk Surabaya) ────
        full["All Stock Cabang"] = full.groupby("No. Barang")["Stock Cabang"].transform("sum")
        full["All SO Cabang"]    = full.groupby("No. Barang")["SO WMA"].transform("sum")

        # ── Add Stock V2 berbasis kategori ─────────────────────────────────
        full["Add Stock"] = calculate_add_stock_v2(full, KAT_COL, "SO WMA", "Stock Cabang")

        # ── Persentase Stock ─────────────────────────
        # SO WMA = 0 & Stock > 0 → 10000 (ada stok tapi tidak ada acuan SO)
        # SO WMA = 0 & Stock = 0 → 0 (keduanya kosong, hasil wajar)
        # SO WMA > 0             → (Stock Cabang / Min Stock) * 100 seperti biasa
        full["Persentase Stock"] = np.where(
            full["Min Stock"] > 0,
            (full["Stock Cabang"] / full["Min Stock"]) * 100,
            np.where(full["Stock Cabang"] > 0, 10000, 0)
        ).round(1)

        # ── Suggested PO V2 (3 skenario) ──────────────────────────────────────
        # calculate_suggested_po_v2 menggunakan Min Stock & Stock Cabang langsung
        full["Suggested PO"] = calculate_suggested_po_v2(full)

        # ── Sisa Stock Surabaya (per SKU) ─────────────────────────
        # Ambil data Surabaya per SKU
        sby = full[full["City"] == "SURABAYA"][["No. Barang", "Stock Cabang", "Min Stock"]].copy()

        sby["Sisa Stock Surabaya"] = np.maximum(
            0,
            sby["Stock Cabang"] - sby["Min Stock"]
        )

        # Merge ke semua cabang
        full = pd.merge(
            full,
            sby[["No. Barang", "Sisa Stock Surabaya"]],
            on="No. Barang",
            how="left"
        )

        full["Sisa Stock Surabaya"] = full["Sisa Stock Surabaya"].fillna(0)

        # ── Pembulatan ─────────────────────────────────────────────────────────
        int_cols = [
            "Stock Cabang", "Min Stock", "Max Stock", "Add Stock",
            "Suggested PO",
            "SO WMA", "SO Mean", "Penjualan Bln 1", "Penjualan Bln 2", "Penjualan Bln 3",
        ] + bulan_columns_renamed
        for col in int_cols:
            if col in full.columns:
                full[col] = pd.to_numeric(full[col], errors="coerce").fillna(0).round(0).astype(int)

        if "Persentase Stock" in full.columns:
            full["Persentase Stock"] = full["Persentase Stock"].fillna(0)

        for col in ["Log (10) WMA", "Avg Log WMA", "Ratio Log WMA"]:
            if col in full.columns:
                full[col] = full[col].round(2)

        st.session_state["stock_v2_result"]     = full.copy()
        st.session_state["stock_v2_bulan_cols"] = bulan_columns_renamed
        st.success("✅ Analisis Stok V2 berhasil dijalankan!")


# ── Render Hasil ───────────────────────────────────────────────────────────────
def _render_results_v2():
    result     = st.session_state["stock_v2_result"].copy()
    result     = result[result["City"] != "OTHERS"]
    bulan_cols = st.session_state.get("stock_v2_bulan_cols", [])
    KAT_COL    = "Kategori ABC (Log-Benchmark - WMA)"

    st.markdown("---")
    st.header("Filter Produk")
    col_f1, col_f2, col_f3 = st.columns(3)
    sel_kat   = col_f1.multiselect("Kategori:",    sorted(result["Kategori Barang"].dropna().unique().astype(str)), key="v2_kat")
    sel_brand = col_f2.multiselect("Brand:",       sorted(result["BRAND Barang"].dropna().unique().astype(str)),   key="v2_brand")
    sel_prod  = col_f3.multiselect("Nama Produk:", sorted(result["Nama Barang"].dropna().unique().astype(str)),    key="v2_prod")

    if sel_kat:   result = result[result["Kategori Barang"].astype(str).isin(sel_kat)]
    if sel_brand: result = result[result["BRAND Barang"].astype(str).isin(sel_brand)]
    if sel_prod:  result = result[result["Nama Barang"].astype(str).isin(sel_prod)]

    st.header("Filter Hasil (Tabel per Kota)")
    col_h1, col_h2 = st.columns(2)
    sel_abc    = col_h1.multiselect("Kategori ABC:", sorted(result[KAT_COL].dropna().unique().astype(str)), key="v2_abc")
    sel_status = col_h2.multiselect("Status Stock:", sorted(result["Status Stock"].dropna().unique().astype(str)), key="v2_status")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Tabel per Kota", "Tabel Gabungan", "Dashboard"])

    with tab1:
        _render_per_kota(result, bulan_cols, sel_abc, sel_status, KAT_COL)
    with tab2:
        _render_pivot_v2(result, bulan_cols, KAT_COL)
    with tab3:
        _render_dashboard_v2(result, KAT_COL)

    # Download
    st.markdown("---")
    st.header("💾 Unduh Hasil Analisis Stock V2")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if "stock_v2_pivot_df" in st.session_state and not st.session_state["stock_v2_pivot_df"].empty:
            st.session_state["stock_v2_pivot_df"].to_excel(writer, sheet_name="All Cities Pivot", index=False)
        result.to_excel(writer, sheet_name="Filtered Data", index=False)
        for city in sorted(result["City"].unique()):
            city_df = result[result["City"] == city]
            if not city_df.empty:
                city_df.to_excel(writer, sheet_name=city[:31], index=False)
    st.download_button(
        "📥 Unduh Excel",
        data=output.getvalue(),
        file_name=f"Hasil_Analisis_Stock_V2_{st.session_state.get('stock_v2_end', 'export')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ── Tabel per Kota ─────────────────────────────────────────────────────────────
def _render_per_kota(result, bulan_cols, sel_abc, sel_status, KAT_COL):
    st.header("Hasil Analisis Stok per Kota")
    KEYS         = ["No. Barang", "Kategori Barang", "BRAND Barang", "Nama Barang"]
    header_style = {"selector": "th", "props": [("background-color", "#0068c9"),
                                                  ("color", "white"), ("text-align", "center")]}

    for city in sorted(result["City"].unique()):
        with st.expander(f"📍 {city}"):
            city_df = result[result["City"] == city].copy()
            if sel_abc:    city_df = city_df[city_df[KAT_COL].isin(sel_abc)]
            if sel_status: city_df = city_df[city_df["Status Stock"].isin(sel_status)]
            if city_df.empty:
                st.write("Tidak ada data yang cocok dengan filter.")
                continue

            metric_order = (
                bulan_cols
                + ["Penjualan Bln 1", "Penjualan Bln 2", "Penjualan Bln 3"]
                + ["SO WMA", "SO Mean", "SO Total"]
                + ["Log (10) WMA", "Avg Log WMA", "Ratio Log WMA", KAT_COL]
                + ["Min Stock", "Max Stock", "Stock Cabang", "Persentase Stock", "Status Stock", "Add Stock", "Suggested PO"]
            )
            display_cols = KEYS + [c for c in metric_order if c in city_df.columns]
            city_df      = city_df[display_cols]

            fmt  = {}
            skip = set(KEYS)
            for col in city_df.columns:
                if col in skip or not pd.api.types.is_numeric_dtype(city_df[col]):
                    continue
                fmt[col] = "{:.2f}" if any(x in col for x in ["Ratio", "Log", "Avg Log"]) else "{:.0f}"

            st.dataframe(
                city_df.style
                .format(fmt, na_rep="-")
                .apply(lambda x: x.map(highlight_kategori_abc_log), subset=[KAT_COL])
                .apply(lambda x: x.map(highlight_status_stock),     subset=["Status Stock"])
                .set_table_styles([header_style]),
                use_container_width=True,
            )


# ── Tabel Gabungan Pivot ───────────────────────────────────────────────────────
def _render_pivot_v2(result, bulan_cols, KAT_COL):
    st.header("📊 Tabel Gabungan Seluruh Kota")
    with st.spinner("Membuat tabel pivot..."):
        KEYS = ["No. Barang", "Kategori Barang", "BRAND Barang", "Nama Barang"]

        # Hanya tampilkan 2 bulan terakhir dari bulan_cols
        bulan_cols_2 = bulan_cols[-2:] if len(bulan_cols) >= 2 else bulan_cols

        pivot_cols = (
            bulan_cols_2
            + ["Penjualan Bln 1", "Penjualan Bln 2", "Penjualan Bln 3"]
            + ["SO WMA", "SO Total"]
            + [KAT_COL]
            + ["Min Stock", "Max Stock", "Stock Cabang", "Persentase Stock", "Status Stock", "Add Stock", "Suggested PO", "Sisa Stock Surabaya"]
        )
        pivot_existing = [c for c in pivot_cols if c in result.columns]
        pivot = result.pivot_table(index=KEYS, columns="City", values=pivot_existing, aggfunc="first")
        pivot.columns = [f"{lv1}_{lv0}" for lv0, lv1 in pivot.columns]
        pivot.reset_index(inplace=True)

        cities  = sorted(result["City"].unique())
        ordered = [f"{city}_{m}" for city in cities for m in pivot_cols]
        existing_ordered = [c for c in ordered if c in pivot.columns]

        # Hitung summary ALL V2
        summary_v2 = calculate_all_summary_v2(result)
        summary_v2.columns = [
            f"All_{c}" if c not in KEYS else c for c in summary_v2.columns
        ]

        # ABC untuk ALL
        from utils import classify_abc_log_benchmark
        all_abc_input = result.groupby(KEYS, as_index=False).agg({"SO WMA": "sum"})
        all_abc_input.rename(columns={"SO WMA": "Total Kuantitas"}, inplace=True)
        all_abc_input["City"] = "ALL"
        all_classified = classify_abc_log_benchmark(all_abc_input, metric_col="Total Kuantitas")
        all_classified.rename(columns={
            "Kategori ABC (Log-Benchmark - Total Kuantitas)": "All_Kategori ABC All",
        }, inplace=True)

        # Gabung semua ke pivot
        pivot = pd.merge(pivot, summary_v2, on=KEYS, how="left")
        pivot = pd.merge(pivot, all_classified[KEYS + ["All_Kategori ABC All"]], on=KEYS, how="left")

        # Kolom summary final
        final_summary = [
            "All_All_Add_Stock_Cabang",
            "All_All_Add_Stock_Surabaya",
            "All_All_Need_From_Supplier",
            "All_Skenario_Distribusi",
            "All_All_Stock_Cabang",
            "All_All_SO_Cabang",
            "All_All_Suggest_PO",
            "All_Kategori ABC All",
        ]
        final_cols = KEYS + existing_ordered + [c for c in final_summary if c in pivot.columns]
        df_style   = pivot[[c for c in final_cols if c in pivot.columns]].copy()

        num_cols   = [c for c in df_style.columns if c not in KEYS
                      and pd.api.types.is_numeric_dtype(df_style[c])
                      and not any(x in c for x in ["Ratio", "Log", "Avg Log"])]
        float_cols = [c for c in df_style.columns if c not in KEYS
                      and any(x in c for x in ["Ratio", "Log", "Avg Log"])]
        obj_cols   = [c for c in df_style.columns if c not in KEYS
                      and c not in num_cols and c not in float_cols]

        df_style[num_cols]   = df_style[num_cols].fillna(0).astype(int)
        df_style[float_cols] = df_style[float_cols].fillna(0)
        df_style[obj_cols]   = df_style[obj_cols].fillna("-")

        # ── Rename kolom: singkatan kota, format bulan, label ringkas ──────────
        CITY_SHORT = {
            "BALI":     "BALI",
            "JAKARTA":  "JKT",
            "JOGJA":    "JOG",
            "MALANG":   "MLG",
            "SEMARANG": "SMG",
            "SURABAYA": "SBY",
        }

        # Nama bulan Indonesia title-case agar tidak kapital semua
        BULAN_TITLE = {
            "JANUARI": "Jan", "FEBRUARI": "Feb", "MARET": "Mar",
            "APRIL": "Apr", "MEI": "Mei", "JUNI": "Jun",
            "JULI": "Jul", "AGUSTUS": "Agu", "SEPTEMBER": "Sep",
            "OKTOBER": "Okt", "NOVEMBER": "Nov", "DESEMBER": "Des",
        }

        def _rename_col(col):
            # Pisah prefix kota dari nama metrik
            for city_long, city_short in CITY_SHORT.items():
                if col.startswith(city_long + "_"):
                    metric = col[len(city_long) + 1:]
                    # Ganti nama bulan kapital → title-case singkat
                    for bln_upper, bln_title in BULAN_TITLE.items():
                        if metric.startswith(bln_upper + " "):
                            tahun = metric.split(" ", 1)[1]
                            metric = f"{bln_title} {tahun}"
                            break
                    # "Penjualan Bln X" → "Bln X"
                    metric = metric.replace("Penjualan Bln ", "Bln ")
                    # Hapus "(Log-Benchmark - WMA)" dari nama kolom ABC
                    metric = metric.replace(" (Log-Benchmark - WMA)", "")
                    return f"{city_short}_{metric}"
            return col

        df_style.rename(columns=_rename_col, inplace=True)

        # Rename kolom All_ agar lebih readable
        rename_display = {
            "All_All_Add_Stock_Cabang":   "All_Add_Cabang",
            "All_All_Add_Stock_Surabaya": "All_Add_Surabaya",
            "All_All_Need_From_Supplier": "All_Need_Distributor",
            "All_Skenario_Distribusi":    "SBY_Skenario",
            "All_All_Stock_Cabang":       "All_Stock_Cabang",
            "All_All_SO_Cabang":          "All_SO_Cabang",
            "All_All_Suggest_PO":         "Restock?",
        }
        df_style.rename(columns=rename_display, inplace=True)

        # Bersihkan label skenario: hapus angka di depan (e.g. "1 - KURANG" → "KURANG")
        if "SBY_Skenario" in df_style.columns:
            df_style["SBY_Skenario"] = df_style["SBY_Skenario"].str.replace(
                r"^\d+\s*-\s*", "", regex=True
            )

        # ── Susun ulang urutan kolom akhir ─────────────────────────────────────
        # 1. Hapus SBY_Suggested PO dari tampilan
        if "SBY_Suggested PO" in df_style.columns:
            df_style.drop(columns=["SBY_Suggested PO"], inplace=True)

        # 2. Rename SBY_Sisa Stock Surabaya → SBY_Sisa Stock
        if "SBY_Sisa Stock Surabaya" in df_style.columns:
            df_style.rename(columns={"SBY_Sisa Stock Surabaya": "SBY_Sisa Stock"}, inplace=True)

        # 3. Susun ulang kolom: kota non-SBY → SBY (tanpa Suggested PO) → SBY_Add Stock
        #    → SBY_Sisa Stock → SBY_Skenario → All_Add_Cabang → All_Stock_Cabang
        #    → All_SO_Cabang → All_Kategori ABC All → All_Need_Distributor → Restock?
        sby_fixed_tail = ["SBY_Add Stock", "SBY_Sisa Stock"]
        sby_skenario_col = ["SBY_Skenario"]
        all_cols_order = [
            "All_Add_Cabang",
            "All_Stock_Cabang",
            "All_SO_Cabang",
            "All_Kategori ABC All",
            "All_Need_Distributor",
            "Restock?",
        ]
        special_cols = set(sby_fixed_tail + sby_skenario_col + all_cols_order + ["Restock?"])
        non_sby_city_cols = [c for c in df_style.columns
                             if c not in KEYS and not c.startswith("SBY_") and c not in special_cols]
        sby_other_cols = [c for c in df_style.columns
                          if c.startswith("SBY_") and c not in special_cols]
        new_order = (
            KEYS
            + non_sby_city_cols
            + sby_other_cols
            + [c for c in sby_fixed_tail if c in df_style.columns]
            + [c for c in sby_skenario_col if c in df_style.columns]
            + [c for c in all_cols_order if c in df_style.columns]
            + [c for c in df_style.columns if c not in KEYS + non_sby_city_cols + sby_other_cols + sby_fixed_tail + sby_skenario_col + all_cols_order]
        )
        df_style = df_style[[c for c in new_order if c in df_style.columns]]

        # col_cfg: pakai nama kolom setelah semua rename selesai
        col_cfg = {}
        for c in df_style.columns:
            if c in KEYS:
                continue
            if pd.api.types.is_numeric_dtype(df_style[c]) and not any(x in c for x in ["Ratio", "Log", "Avg Log"]):
                col_cfg[c] = st.column_config.NumberColumn(format="%.0f")
            elif any(x in c for x in ["Ratio", "Log", "Avg Log"]):
                col_cfg[c] = st.column_config.NumberColumn(format="%.2f")

        st.session_state["stock_v2_pivot_df"] = df_style.copy()
        st.dataframe(df_style, column_config=col_cfg, use_container_width=True)

        # Tampilkan legenda skenario
        st.markdown("---")
        st.markdown("**Legenda Skenario Distribusi:**")
        col1, col2, col3 = st.columns(3)
        col1.error("**1 - KURANG**\nStok Sisa = 0.\nSurabaya tidak bisa kirim ke manapun.\nSemua cabang PO = 0, tunggu distributor.")
        col2.warning("**2 - SISA**\n0 < Stok Sisa < All Add Cabang.\nAda sisa tapi tidak cukup untuk semua.\nDistribusi berdasarkan urgency (% stok terkecil dahulu).\nTarget: Add Stock / 2 − Stock Cabang per cabang.")
        col3.success("**3 - OVER**\nStok Sisa ≥ All Add Cabang.\nSisa cukup untuk semua cabang.\nSemua cabang dapat Add Stock penuh.")


# ── Dashboard ──────────────────────────────────────────────────────────────────
def _render_dashboard_v2(result, KAT_COL):
    st.header("📈 Dashboard Analisis Stock V2")
    if result.empty:
        st.info("Tidak ada data untuk ditampilkan.")
        return

    # Summary skenario distribusi
    summary_v2 = calculate_all_summary_v2(result)
    if not summary_v2.empty and "Skenario_Distribusi" in summary_v2.columns:
        st.subheader("Distribusi Skenario per Produk")
        skenario_counts = summary_v2["Skenario_Distribusi"].value_counts()
        col_s1, col_s2, col_s3 = st.columns(3)
        cols_map = {
            "1 - KURANG": (col_s1, "🔴"),
            "2 - SISA":   (col_s2, "🟡"),
            "3 - OVER":   (col_s3, "🟢"),
        }
        for label, (col, icon) in cols_map.items():
            count = skenario_counts.get(label, 0)
            col.metric(f"{icon} {label}", f"{count} SKU")
        st.markdown("---")

    # Metric cards
    total_under = result[result["Status Stock"] == "Understock"].shape[0]
    total_over  = result[result["Status Stock"].str.contains("Overstock", na=False)].shape[0]
    total_need  = summary_v2["All_Need_From_Supplier"].sum() if not summary_v2.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Produk Understock",  f"{total_under} SKU")
    col2.metric("Total Produk Overstock",   f"{total_over} SKU")
    col3.metric("Total Need From Supplier", f"{int(total_need)} unit")

    st.markdown("---")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("Distribusi Kategori ABC")
        st.bar_chart(result[KAT_COL].value_counts())
    with col_c2:
        st.subheader("Distribusi Status Stok")
        st.bar_chart(result["Status Stock"].value_counts())

    st.markdown("---")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("Top 5 Produk Paling Understock")
        top_u = (
            result[result["Status Stock"] == "Understock"]
            .sort_values("Add Stock", ascending=False).head(5)
        )
        st.dataframe(top_u[["Nama Barang", "City", "Add Stock", "Stock Cabang", "Min Stock"]],
                     use_container_width=True)
    with col_t2:
        st.subheader("Top 5 Produk Butuh Supplier Terbanyak")
        if not summary_v2.empty:
            top_sup = summary_v2.nlargest(5, "All_Need_From_Supplier")[
                ["Nama Barang", "All_Need_From_Supplier", "All_Add_Stock_Cabang",
                 "Skenario_Distribusi"]
            ]
            st.dataframe(top_sup, use_container_width=True)
