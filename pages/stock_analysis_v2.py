"""
pages/lgbm_analysis.py
─────────────────────────────────────────────────────────────────────────────
Halaman: Analisa Stock V2 + LGBM Residual Learning

Fitur halaman:
    1. Jalankan Analisa V2 (WMA) + prediksi koreksi LGBM sekaligus
    2. Training LGBM dari histori SO (rolling window 30 hari)
    3. Evaluasi model: MAE, RMSE, improvement %, scatter plot, distribusi residual
    4. Insight & narasi otomatis per kota dan kategori ABC
    5. Tabel detail: SO WMA | LGBM Koreksi | SO Final (opsi B — berdampingan)
    6. Download hasil
─────────────────────────────────────────────────────────────────────────────
"""

import math
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import timedelta

from utils import (
    BULAN_INDONESIA, map_nama_dept, map_city, convert_df_to_excel,
    calculate_daily_wma, classify_abc_log_benchmark,
    calculate_min_stock, calculate_max_stock,
    get_status_stock, melt_stock_by_city,
    highlight_kategori_abc_log, highlight_status_stock,
)
from utils.analysis import (
    calculate_add_stock_v2, calculate_suggested_po_v2,
    calculate_all_summary_v2, calculate_persentase_stock,
)
from utils.gdrive import init_drive_service, FOLDER_HASIL_ANALISIS
from utils.lgbm_predictor import (
    build_training_dataset, train_lgbm_model,
    predict_correction, save_model_to_gdrive, load_model_from_gdrive,
    check_model_exists_in_gdrive, summarize_metrics, EXCLUDE_CATEGORIES,
)

KAT_COL = "Kategori ABC (Log-Benchmark - WMA)"


# ══════════════════════════════════════════════════════════════════════════════
# RENDER UTAMA
# ══════════════════════════════════════════════════════════════════════════════

def render():
    st.title("🤖 Analisa Stock V2 + LGBM Residual Learning")

    st.info("""
    **Arsitektur Residual Learning:**
    ```
    SO_Final  =  SO_WMA  +  LGBM_Koreksi(error WMA)
    ```
    LightGBM mempelajari pola *error* yang dihasilkan WMA berdasarkan histori rolling window 30 hari,
    lalu mengoreksinya. Kategori **F & E** tetap menggunakan WMA murni.
    """)

    # Cek prerequisite
    if (st.session_state.get("df_penjualan", pd.DataFrame()).empty or
            st.session_state.get("produk_ref", pd.DataFrame()).empty or
            st.session_state.get("df_stock", pd.DataFrame()).empty):
        st.warning("⚠️ Harap muat semua data di halaman **Input Data** terlebih dahulu.")
        st.stop()

    drive_service, drive_ok = init_drive_service()

    # ── Sidebar konfigurasi ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Konfigurasi LGBM")
        test_ratio = st.slider("Proporsi Test (kronologis)", 0.15, 0.40, 0.30, 0.05)
        step_days  = st.selectbox("Step rolling window (hari)", [30, 15], index=0)

    # ── Preprocessing penjualan ───────────────────────────────────────────────
    penjualan  = st.session_state.df_penjualan.copy()
    produk_ref = st.session_state.produk_ref.copy()
    df_stock   = st.session_state.df_stock.copy()

    for df in [penjualan, produk_ref, df_stock]:
        if "No. Barang" in df.columns:
            df["No. Barang"] = df["No. Barang"].astype(str).str.strip()

    penjualan.rename(columns={"Qty": "Kuantitas"}, inplace=True, errors="ignore")
    penjualan["Nama Dept"] = penjualan.apply(map_nama_dept, axis=1)
    penjualan["City"]      = penjualan["Nama Dept"].apply(map_city)
    produk_ref.rename(columns={"Keterangan Barang": "Nama Barang"}, inplace=True, errors="ignore")
    if "Kategori Barang" in produk_ref.columns:
        produk_ref["Kategori Barang"] = produk_ref["Kategori Barang"].astype(str).str.strip().str.upper()
    penjualan["City"]       = penjualan["City"].astype(str).str.strip().str.upper()
    penjualan["Tgl Faktur"] = pd.to_datetime(penjualan["Tgl Faktur"], errors="coerce")

    st.markdown("---")

    # ── Tanggal analisis ──────────────────────────────────────────────────────
    default_end = penjualan["Tgl Faktur"].dropna().max().date()
    if st.session_state.stock_filename:
        m = re.search(r"(\d{8})", st.session_state.stock_filename)
        if m:
            try:
                from datetime import datetime
                default_end = datetime.strptime(m.group(1), "%d%m%Y").date()
            except ValueError:
                pass

    col1, col2 = st.columns(2)
    col1.date_input("Tanggal Awal (90 Hari)", value=default_end - timedelta(days=89),
                    key="lgbm_start", disabled=True)
    end_date = col2.date_input("Tanggal Akhir", value=default_end, key="lgbm_end")

    # ── Tombol aksi ───────────────────────────────────────────────────────────
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)

    btn_train   = col_a.button("🧠 Training LGBM", type="primary",  use_container_width=True,
                                help="Build rolling window dataset & latih model LGBM")
    btn_predict = col_b.button("🔮 Terapkan Koreksi LGBM", use_container_width=True,
                                help="Jalankan analisis WMA lalu terapkan koreksi LGBM")
    btn_clear   = col_c.button("🗑️ Reset Hasil", use_container_width=True)

    if btn_clear:
        for k in ["lgbm_train_result", "lgbm_train_dataset", "lgbm_model_sess",
                  "lgbm_enc_sess", "lgbm_predict_result", "lgbm_v2_base"]:
            st.session_state.pop(k, None)
        st.rerun()

    # ── Training ──────────────────────────────────────────────────────────────
    if btn_train:
        _run_training(penjualan, produk_ref, drive_service, drive_ok, test_ratio, step_days)

    # ── Prediksi ──────────────────────────────────────────────────────────────
    if btn_predict:
        _run_predict(penjualan, produk_ref, df_stock, end_date, drive_service, drive_ok)

    # ── Tampilkan hasil ───────────────────────────────────────────────────────
    if st.session_state.get("lgbm_train_result"):
        _render_training_section()

    if st.session_state.get("lgbm_predict_result") is not None:
        _render_predict_section(end_date)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def _run_training(penjualan, produk_ref, drive_service, drive_ok, test_ratio, step_days):
    df_so = penjualan.copy()
    # Butuh ABC reference — coba dari hasil V2 yang tersimpan, atau jalankan dulu
    df_abc = st.session_state.get("lgbm_v2_base") or st.session_state.get("stock_v2_result")
    if df_abc is None:
        st.warning("Jalankan **Terapkan Koreksi LGBM** sekali dulu untuk membuat referensi ABC, "
                   "lalu klik **Training LGBM**.")
        return

    prog = st.progress(0, "Membangun rolling window dataset...")
    try:
        df_train = build_training_dataset(df_so, df_abc, step_days=step_days)
        if df_train.empty:
            st.error("Dataset training kosong. Pastikan data SO mencukupi (minimal 4 bulan).")
            prog.empty()
            return

        n_sampel  = len(df_train)
        n_windows = df_train["window_end"].nunique()
        prog.progress(35, f"Dataset: {n_sampel:,} sampel | {n_windows} window. Training LGBM...")

        result = train_lgbm_model(df_train, test_ratio=test_ratio)
        prog.progress(80, "Training selesai. Menyimpan ke Google Drive...")

        save_ok = False
        if drive_ok:
            save_ok = save_model_to_gdrive(
                result["model"], result["encoders"], drive_service, FOLDER_HASIL_ANALISIS)

        prog.progress(100, "✅ Selesai!" if save_ok else "⚠️ Selesai (gagal simpan ke Drive)")

        st.session_state["lgbm_train_result"]  = result
        st.session_state["lgbm_train_dataset"] = df_train
        st.session_state["lgbm_model_sess"]    = result["model"]
        st.session_state["lgbm_enc_sess"]      = result["encoders"]

        if save_ok:
            st.success(f"✅ Model disimpan ke Google Drive | "
                       f"{n_sampel:,} sampel | {n_windows} window | "
                       f"Best iter: {result['model'].best_iteration_}")
        else:
            st.warning("⚠️ Training selesai tapi model gagal disimpan ke Drive. "
                       "Koreksi masih bisa diterapkan untuk sesi ini.")

    except ImportError as e:
        prog.empty()
        st.error(f"❌ {e}")
        st.code("lightgbm\nscikit-learn\njoblib", language="text")
        st.info("Tambahkan baris di atas ke `requirements.txt` lalu restart app.")
    except Exception as e:
        prog.empty()
        st.error(f"❌ Error training: {e}")
        import traceback
        with st.expander("Detail error"):
            st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# ANALISIS V2 + PREDIKSI LGBM
# ══════════════════════════════════════════════════════════════════════════════

def _run_predict(penjualan, produk_ref, df_stock, end_date, drive_service, drive_ok):
    # Load model
    model    = st.session_state.get("lgbm_model_sess")
    encoders = st.session_state.get("lgbm_enc_sess")

    if model is None:
        if not drive_ok:
            st.error("Google Drive tidak terhubung dan model belum ada di sesi ini. "
                     "Jalankan Training terlebih dahulu.")
            return
        with st.spinner("Mengunduh model dari Google Drive..."):
            model, encoders = load_model_from_gdrive(drive_service, FOLDER_HASIL_ANALISIS)
        if model is None:
            st.warning("Model belum tersedia di Google Drive. "
                       "Analisis WMA akan tetap dijalankan tanpa koreksi LGBM.")
        else:
            st.session_state["lgbm_model_sess"] = model
            st.session_state["lgbm_enc_sess"]   = encoders

    with st.spinner("Menjalankan Analisa Stock V2 + Koreksi LGBM..."):
        full = _run_wma_analysis(penjualan, produk_ref, df_stock, end_date)
        if full is None:
            return

        # Simpan base result (tanpa LGBM) sebagai referensi ABC untuk training
        st.session_state["lgbm_v2_base"] = full.copy()

        # Terapkan koreksi LGBM jika model tersedia
        if model is not None and encoders is not None:
            full = predict_correction(full, model, encoders, penjualan, end_date)
            n_corrected = full["LGBM_Available"].sum()
            st.success(f"✅ Analisis selesai | Koreksi LGBM diterapkan pada "
                       f"**{n_corrected:,}** dari {len(full):,} baris "
                       f"(F & E menggunakan WMA murni)")
        else:
            full["LGBM_Koreksi"]  = 0.0
            full["SO_Final"]       = full["SO WMA"]
            full["LGBM_Available"] = False
            st.info("ℹ️ Analisis WMA berhasil. Koreksi LGBM tidak diterapkan (model belum dilatih).")

        st.session_state["lgbm_predict_result"] = full


def _run_wma_analysis(penjualan, produk_ref, df_stock, end_date) -> pd.DataFrame | None:
    """Jalankan analisis WMA V2 dan return DataFrame lengkap."""
    end_dt    = pd.to_datetime(end_date)
    wma_start = end_dt - pd.DateOffset(days=89)
    penj_90   = penjualan[penjualan["Tgl Faktur"].between(wma_start, end_dt)]

    if penj_90.empty:
        st.error("Tidak ada data penjualan dalam 90 hari terakhir.")
        return None

    r1_end, r1_start = end_dt, end_dt - pd.DateOffset(days=29)
    r2_end, r2_start = end_dt - pd.DateOffset(days=30), end_dt - pd.DateOffset(days=59)
    r3_end, r3_start = end_dt - pd.DateOffset(days=60), end_dt - pd.DateOffset(days=89)

    def _sales(s, e, col):
        df = (penj_90[penj_90["Tgl Faktur"].between(s, e)]
              .groupby(["City", "No. Barang"])["Kuantitas"].sum().reset_index())
        df.rename(columns={"Kuantitas": col}, inplace=True)
        return df

    sales_m1 = _sales(r1_start, r1_end, "Penjualan Bln 1")
    sales_m2 = _sales(r2_start, r2_end, "Penjualan Bln 2")
    sales_m3 = _sales(r3_start, r3_end, "Penjualan Bln 3")
    _t90 = penj_90.groupby(["City", "No. Barang"])["Kuantitas"].sum().reset_index()
    total_90 = _t90[["City", "No. Barang"]].copy()
    total_90["AVG Mean"] = (_t90["Kuantitas"] / 3).values

    _grp = ["City", "No. Barang"]
    _end = pd.to_datetime(end_date)
    _r1s = _end - pd.DateOffset(days=29)
    _r2s, _r2e = _end - pd.DateOffset(days=59), _end - pd.DateOffset(days=30)
    _r3s, _r3e = _end - pd.DateOffset(days=89), _end - pd.DateOffset(days=60)
    _s1 = penj_90[penj_90["Tgl Faktur"].between(_r1s, _end)].groupby(_grp)["Kuantitas"].sum().rename("s1").reset_index()
    _s2 = penj_90[penj_90["Tgl Faktur"].between(_r2s, _r2e)].groupby(_grp)["Kuantitas"].sum().rename("s2").reset_index()
    _s3 = penj_90[penj_90["Tgl Faktur"].between(_r3s, _r3e)].groupby(_grp)["Kuantitas"].sum().rename("s3").reset_index()
    wma_grouped = _s1.merge(_s2, on=_grp, how="outer").merge(_s3, on=_grp, how="outer").fillna(0)
    wma_grouped["AVG WMA"] = (wma_grouped["s1"]*0.5 + wma_grouped["s2"]*0.3 + wma_grouped["s3"]*0.2).apply(math.ceil)
    wma_grouped.drop(columns=["s1","s2","s3"], inplace=True)

    barang_list = produk_ref[["No. Barang", "Kategori Barang", "BRAND Barang", "Nama Barang"]].drop_duplicates()
    city_list   = penjualan["City"].unique() if "City" in penjualan.columns else penj_90["City"].unique()
    kombinasi   = pd.MultiIndex.from_product(
        [city_list, barang_list["No. Barang"]], names=["City", "No. Barang"]).to_frame(index=False)
    full = pd.merge(kombinasi, barang_list, on="No. Barang", how="left")

    for dm in [wma_grouped, sales_m1, sales_m2, sales_m3, total_90]:
        full = pd.merge(full, dm, on=["City", "No. Barang"], how="left")

    penj_90c = penj_90.copy()
    penj_90c["Bulan"] = penj_90c["Tgl Faktur"].dt.to_period("M")
    monthly = (penj_90c.groupby(["City", "No. Barang", "Bulan"])["Kuantitas"]
               .sum().unstack(fill_value=0).reset_index())
    full = pd.merge(full, monthly, on=["City", "No. Barang"], how="left")

    for col in full.columns:
        try:
            if pd.api.types.is_numeric_dtype(full[col]):
                full[col] = full[col].fillna(0)
            else:
                full[col] = full[col].astype(str).fillna("")
        except Exception:
            pass

    period_cols = sorted([c for c in full.columns if isinstance(c, pd.Period)])
    rename_map  = {c: f"{BULAN_INDONESIA[c.month]} {c.year}" for c in period_cols}
    full.rename(columns=rename_map, inplace=True)
    full.rename(columns={"AVG WMA": "SO WMA", "AVG Mean": "SO Mean"}, inplace=True)
    full["SO Total"] = full["SO WMA"]

    log_df = classify_abc_log_benchmark(full.copy(), metric_col="SO WMA")
    full   = pd.merge(full, log_df[["City", "No. Barang", KAT_COL,
                                    "Ratio Log WMA", "Log (10) WMA", "Avg Log WMA"]],
                      on=["City", "No. Barang"], how="left")

    full["Min Stock"] = calculate_min_stock(full, KAT_COL, "SO WMA")
    full["Max Stock"] = calculate_max_stock(full, KAT_COL, "SO WMA")

    stock_melted = melt_stock_by_city(df_stock.rename(columns=lambda x: x.strip()))
    full = pd.merge(full, stock_melted, on=["City", "No. Barang"], how="left").rename(
        columns={"Stock": "Stock Cabang"})
    full["Stock Cabang"] = full["Stock Cabang"].fillna(0)
    full["Status Stock"]  = full.apply(get_status_stock, axis=1)
    full["All Stock Cabang"] = full.groupby("No. Barang")["Stock Cabang"].transform("sum")
    full["All SO Cabang"]    = full.groupby("No. Barang")["SO WMA"].transform("sum")
    full["Add Stock"]        = calculate_add_stock_v2(full, KAT_COL, "SO WMA", "Stock Cabang")
    full["Persentase Stock"]  = calculate_persentase_stock(full)
    full["Suggested PO"]     = calculate_suggested_po_v2(full)

    sby = full[full["City"] == "SURABAYA"][["No. Barang", "Stock Cabang", "Min Stock"]].copy()
    sby["Sisa Stock Surabaya"] = np.maximum(0, sby["Stock Cabang"] - sby["Min Stock"])
    full = pd.merge(full, sby[["No. Barang", "Sisa Stock Surabaya"]], on="No. Barang", how="left")
    full["Sisa Stock Surabaya"] = full["Sisa Stock Surabaya"].fillna(0)

    return full


# ══════════════════════════════════════════════════════════════════════════════
# RENDER TRAINING SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_training_section():
    result = st.session_state["lgbm_train_result"]
    df_train = st.session_state.get("lgbm_train_dataset", pd.DataFrame())

    st.markdown("---")
    st.header("🧠 Hasil Training Model LGBM")

    # ── Narasi Training ───────────────────────────────────────────────────────
    m_test = result["metrics"]["test"]
    m_train = result["metrics"]["train"]
    imp     = m_test["improvement_mae_pct"]
    direction = "menurun" if imp > 0 else "meningkat"
    quality   = "baik" if imp > 5 else ("cukup baik" if imp > 0 else "belum optimal")

    st.markdown(f"""
    > **Ringkasan Training:** Model LGBM dilatih menggunakan **{result['n_windows']} rolling window**
    > dengan total **{m_train['n_samples']:,} sampel training** dan **{m_test['n_samples']:,} sampel test**.
    > Evaluasi pada data test menunjukkan MAE **{direction}** dari `{m_test['mae_wma']:.2f}` (WMA murni)
    > menjadi `{m_test['mae_final']:.2f}` (WMA + LGBM), atau **{abs(imp):.1f}% {direction}** —
    > performa model terbilang **{quality}**.
    """)

    # ── Metrik evaluasi ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Tabel Evaluasi")
        metrics_df = summarize_metrics(result["metrics"])
        st.dataframe(
            metrics_df.style.format({
                "MAE — WMA": "{:.4f}", "MAE — SO Final": "{:.4f}",
                "RMSE — WMA": "{:.4f}", "RMSE — SO Final": "{:.4f}",
                "Improvement MAE (%)": "{:+.2f}%",
            }).background_gradient(subset=["Improvement MAE (%)"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True,
        )

        # Penjelasan metrik
        with st.expander("ℹ️ Cara membaca metrik"):
            st.markdown("""
            | Metrik | Penjelasan |
            |---|---|
            | **MAE** | Rata-rata selisih absolut prediksi vs aktual (unit: pcs). Semakin kecil semakin baik. |
            | **RMSE** | Sama seperti MAE tapi lebih sensitif terhadap error besar. |
            | **Improvement MAE (%)** | Seberapa besar LGBM mengurangi error WMA. Positif = lebih baik. |
            | **TRAIN** | Evaluasi pada data 70% pertama (kronologis). |
            | **TEST** | Evaluasi pada 30% data terbaru — ini yang paling penting. |
            """)

    with col2:
        st.subheader("📈 Feature Importance")
        fi = result["feature_importance"]

        label_map = {
            "so_wma": "SO WMA", "penjualan_bln1": "Penjualan Bln 1",
            "penjualan_bln2": "Penjualan Bln 2", "penjualan_bln3": "Penjualan Bln 3",
            "kategori_abc_enc": "Kategori ABC", "kategori_brg_enc": "Kategori Barang",
            "brand_enc": "Brand", "hari_ke": "Hari Ke (Temporal)",
            "bulan": "Bulan", "city_enc": "Kota",
            "residual_lag1": "Residual Lag 1 (30h)", "residual_lag2": "Residual Lag 2 (60h)",
        }
        fi.index = [label_map.get(i, i) for i in fi.index]
        fi_sorted = fi.sort_values()

        colors = ["#4C9BE8" if v > fi.median() else "#B0C4DE" for v in fi_sorted]
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.barh(fi_sorted.index, fi_sorted.values, color=colors)
        ax.set_xlabel("Importance Score", fontsize=9)
        ax.set_title("Kontribusi Fitur terhadap Model LGBM", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        for bar, val in zip(bars, fi_sorted.values):
            ax.text(val + fi_sorted.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:,.0f}", va="center", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Narasi feature importance
        top_feat = fi.idxmax()
        st.caption(f"💡 Fitur paling berpengaruh: **{top_feat}**. "
                   f"Ini berarti LGBM paling banyak belajar dari variabel tersebut "
                   f"untuk memprediksi seberapa besar error WMA.")

    # ── Scatter Plot: WMA vs Final vs Actual ─────────────────────────────────
    st.subheader("🎯 Scatter Plot: Prediksi vs Aktual (Data Test)")
    df_test = result.get("df_test", pd.DataFrame())
    if not df_test.empty:
        fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

        # WMA vs Actual
        ax = axes[0]
        ax.scatter(df_test["actual_so"], df_test["so_wma"],
                   alpha=0.3, s=8, color="#4C9BE8", label="SO WMA")
        lim = max(df_test["actual_so"].max(), df_test["so_wma"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=1, label="Ideal")
        ax.set_xlabel("Aktual SO", fontsize=9)
        ax.set_ylabel("Prediksi WMA", fontsize=9)
        ax.set_title(f"WMA Murni\nMAE = {m_test['mae_wma']:.2f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

        # Final vs Actual
        ax = axes[1]
        ax.scatter(df_test["actual_so"], df_test["so_final"],
                   alpha=0.3, s=8, color="#F4845F", label="SO Final")
        ax.plot([0, lim], [0, lim], "r--", lw=1, label="Ideal")
        ax.set_xlabel("Aktual SO", fontsize=9)
        ax.set_ylabel("Prediksi SO Final", fontsize=9)
        ax.set_title(f"WMA + LGBM\nMAE = {m_test['mae_final']:.2f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

        plt.suptitle("Perbandingan Akurasi Prediksi pada Data Test", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ── Distribusi Residual ───────────────────────────────────────────────
        st.subheader("📉 Distribusi Residual")
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 3.5))

        res_wma   = df_test["actual_so"] - df_test["so_wma"]
        res_final = df_test["actual_so"] - df_test["so_final"]

        for ax, res, label, color in [
            (axes3[0], res_wma,   "WMA Murni",    "#4C9BE8"),
            (axes3[1], res_final, "WMA + LGBM",   "#F4845F"),
        ]:
            ax.hist(res, bins=50, color=color, alpha=0.75, edgecolor="white")
            ax.axvline(0, color="red", linestyle="--", lw=1.5, label="Ideal (0)")
            ax.axvline(res.mean(), color="darkblue", linestyle="-", lw=1.5,
                       label=f"Mean = {res.mean():.2f}")
            ax.set_xlabel("Residual (Aktual − Prediksi)", fontsize=9)
            ax.set_ylabel("Frekuensi", fontsize=9)
            ax.set_title(f"Distribusi Residual — {label}", fontsize=10)
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=8)

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        st.caption(
            "💡 **Interpretasi:** Distribusi residual yang mendekati 0 (garis merah) "
            "dan simetris menunjukkan model tidak bias. Jika distribusi WMA + LGBM "
            "lebih sempit dari WMA murni, berarti LGBM berhasil mengurangi error."
        )

    # ── Dataset training info ─────────────────────────────────────────────────
    if not df_train.empty:
        with st.expander("🗂️ Info Dataset Training"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Sampel", f"{len(df_train):,}")
            c2.metric("Jumlah Window", f"{df_train['window_end'].nunique()}")
            c3.metric("Produk Unik", f"{df_train['No. Barang'].nunique():,}")
            c4.metric("Kota", f"{df_train['City'].nunique()}")

            st.markdown("**Distribusi sampel per Kategori ABC:**")
            kat_dist = df_train["kategori_abc"].value_counts().reset_index()
            kat_dist.columns = ["Kategori", "Jumlah Sampel"]
            st.dataframe(kat_dist, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# RENDER PREDIKSI SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_predict_section(end_date):
    df = st.session_state["lgbm_predict_result"].copy()
    df = df[df["City"] != "OTHERS"]
    has_lgbm = df["LGBM_Available"].any() if "LGBM_Available" in df.columns else False

    st.markdown("---")
    st.header("📊 Hasil Analisis & Koreksi LGBM")

    # ── Narasi & Insight Otomatis ─────────────────────────────────────────────
    _render_insights(df, has_lgbm, end_date)

    st.markdown("---")

    # ── Perbandingan SO WMA vs SO Final ───────────────────────────────────────
    if has_lgbm:
        st.subheader("🔁 Dampak Koreksi LGBM")
        df_elig = df[df["LGBM_Available"]]

        c1, c2, c3, c4 = st.columns(4)
        avg_wma    = df_elig["SO WMA"].mean()
        avg_final  = df_elig["SO_Final"].mean()
        avg_kor    = df_elig["LGBM_Koreksi"].mean()
        pct_change = (avg_final - avg_wma) / avg_wma * 100 if avg_wma > 0 else 0
        c1.metric("Rata-rata SO WMA",    f"{avg_wma:.1f}")
        c2.metric("Rata-rata SO Final",  f"{avg_final:.1f}", f"{pct_change:+.1f}%")
        c3.metric("Rata-rata Koreksi",   f"{avg_kor:+.2f}")
        c4.metric("Baris Dikoreksi",     f"{len(df_elig):,}")

        # Chart perbandingan per kota
        city_comp = (df_elig.groupby("City")
                     .agg(WMA=("SO WMA", "sum"), Final=("SO_Final", "sum"))
                     .reset_index().sort_values("WMA", ascending=False))

        fig, ax = plt.subplots(figsize=(9, 3.5))
        x     = np.arange(len(city_comp))
        w     = 0.35
        ax.bar(x - w/2, city_comp["WMA"],   w, label="SO WMA",   color="#4C9BE8", alpha=0.85)
        ax.bar(x + w/2, city_comp["Final"], w, label="SO Final", color="#F4845F", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(city_comp["City"], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Total SO (unit)", fontsize=9)
        ax.set_title("SO WMA vs SO Final per Kota", fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ── Tabel Detail ──────────────────────────────────────────────────────────
    st.subheader("🔍 Tabel Detail per Produk × Kota")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    sel_city  = col_f1.multiselect("Kota:",          sorted(df["City"].dropna().unique()),                   key="lg_city")
    sel_kat   = col_f2.multiselect("Kategori:",      sorted(df["Kategori Barang"].dropna().unique().astype(str)), key="lg_kat")
    sel_abc   = col_f3.multiselect("ABC:",           sorted(df[KAT_COL].dropna().unique()),                  key="lg_abc")
    sel_status= col_f4.multiselect("Status Stock:",  sorted(df["Status Stock"].dropna().unique()),            key="lg_status")

    df_view = df.copy()
    if sel_city:   df_view = df_view[df_view["City"].isin(sel_city)]
    if sel_kat:    df_view = df_view[df_view["Kategori Barang"].astype(str).isin(sel_kat)]
    if sel_abc:    df_view = df_view[df_view[KAT_COL].isin(sel_abc)]
    if sel_status: df_view = df_view[df_view["Status Stock"].isin(sel_status)]

    base_cols = ["No. Barang", "Nama Barang", "BRAND Barang", "Kategori Barang", "City", KAT_COL]
    lgbm_cols = ["SO WMA", "LGBM_Koreksi", "SO_Final"] if has_lgbm else ["SO WMA"]
    stock_cols = ["Min Stock", "Max Stock", "Stock Cabang", "Add Stock", "Suggested PO",
                  "Status Stock", "Persentase Stock"]
    show_cols  = [c for c in base_cols + lgbm_cols + stock_cols if c in df_view.columns]

    def _style_row(row):
        styles = [""] * len(row)
        if "SO_Final" in row.index and "SO WMA" in row.index:
            idx_final = list(row.index).index("SO_Final")
            if row["SO_Final"] > row["SO WMA"]:
                styles[idx_final] = "background-color: #d4edda; color: #155724"  # hijau
            elif row["SO_Final"] < row["SO WMA"]:
                styles[idx_final] = "background-color: #fff3cd; color: #856404"  # kuning
        return styles

    styled = df_view[show_cols].style.apply(_style_row, axis=1)
    if "LGBM_Koreksi" in show_cols:
        styled = styled.format({"LGBM_Koreksi": "{:+.2f}", "Persentase Stock": "{:.1f}%"}, na_rep="-")

    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(f"Menampilkan {len(df_view):,} baris | "
               f"🟢 SO Final > WMA (LGBM koreksi naik) | 🟡 SO Final < WMA (LGBM koreksi turun)")

    # ── Legenda warna ─────────────────────────────────────────────────────────
    with st.expander("ℹ️ Cara Membaca Tabel"):
        st.markdown("""
        | Kolom | Penjelasan |
        |---|---|
        | **SO WMA** | Prediksi demand dari Weighted Moving Average (metode existing, 90 hari histori) |
        | **LGBM_Koreksi** | Koreksi residual dari LGBM. **Positif** = WMA terlalu rendah. **Negatif** = WMA terlalu tinggi. |
        | **SO_Final** | Prediksi akhir = `ceil(SO WMA + LGBM_Koreksi)`, min 0. Ini yang direkomendasikan sebagai acuan. |
        | **Warna SO_Final** | 🟢 Hijau = LGBM naikkan prediksi | 🟡 Kuning = LGBM turunkan prediksi | Putih = tidak berubah |
        | **Min/Max/Add Stock** | Masih berbasis SO WMA (perlu re-run analisis dengan SO_Final jika mau dipakai penuh) |
        """)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ Download Hasil Lengkap (.xlsx)",
            data=convert_df_to_excel(df_view[show_cols]),
            file_name="hasil_lgbm_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # Summary per kota
    summary_cols = ["City", "SO WMA", "SO_Final", "Stock Cabang",
                    "Min Stock", "Add Stock", "Suggested PO"]
    summary_cols = [c for c in summary_cols if c in df_view.columns]
    df_summary   = df_view.groupby("City")[summary_cols[1:]].sum().reset_index()

    with col_dl2:
        st.download_button(
            "⬇️ Download Summary per Kota (.xlsx)",
            data=convert_df_to_excel(df_summary),
            file_name="summary_per_kota_lgbm.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# INSIGHT & NARASI OTOMATIS
# ══════════════════════════════════════════════════════════════════════════════

def _render_insights(df: pd.DataFrame, has_lgbm: bool, end_date):
    st.subheader("💡 Insight & Narasi Otomatis")

    df_w = df[df["City"] != "OTHERS"].copy()
    df_elig = df_w[df_w.get("LGBM_Available", pd.Series(False, index=df_w.index))] if has_lgbm else df_w

    # ── 1. Insight Status Stock Global ───────────────────────────────────────
    if "Status Stock" in df_w.columns:
        status_counts = df_w["Status Stock"].value_counts()
        total = len(df_w)
        n_under = status_counts.get("Understock", 0)
        n_over  = status_counts.get("Overstock",  0)
        n_bal   = status_counts.get("Balance",    0)

        pct_under = n_under / total * 100 if total > 0 else 0
        pct_over  = n_over  / total * 100 if total > 0 else 0
        pct_bal   = n_bal   / total * 100 if total > 0 else 0

        # Tone narasi
        if pct_under > 30:
            kondisi = "perlu perhatian — cukup banyak posisi understock"
            rekomendasi = "Prioritaskan pengiriman ke cabang dengan persentase stok terendah."
        elif pct_over > 30:
            kondisi = "cukup sehat stoknya, namun ada potensi overstock"
            rekomendasi = "Pertimbangkan redistribusi atau tahan PO untuk SKU yang overstock."
        else:
            kondisi = "relatif seimbang"
            rekomendasi = "Lanjutkan monitoring rutin dan pastikan produk kategori A & B selalu balance."

        st.markdown(f"""
        #### 📌 Kondisi Stok per {end_date}

        Secara keseluruhan, kondisi stok **{kondisi}**.
        Dari **{total:,} kombinasi produk × kota** yang dianalisis:
        - 🔴 **Understock:** {n_under:,} ({pct_under:.1f}%)
        - 🟢 **Balance:** {n_bal:,} ({pct_bal:.1f}%)
        - 🟠 **Overstock:** {n_over:,} ({pct_over:.1f}%)

        _{rekomendasi}_
        """)

    # ── 2. Insight per Kota ───────────────────────────────────────────────────
    if "City" in df_w.columns and "Status Stock" in df_w.columns:
        st.markdown("#### 🏙️ Kondisi per Kota")
        city_stat = (df_w.groupby(["City", "Status Stock"])
                     .size().unstack(fill_value=0).reset_index())

        fig, ax = plt.subplots(figsize=(10, 3.5))
        status_order = ["Understock", "Balance", "Overstock", "Overstock F"]
        colors_map   = {"Understock": "#FFC107", "Balance": "#28A745",
                        "Overstock": "#FF7043", "Overstock F": "#EF5350"}
        bottom = np.zeros(len(city_stat))
        for status in status_order:
            if status in city_stat.columns:
                vals = city_stat[status].values
                ax.bar(city_stat["City"], vals, bottom=bottom,
                       label=status, color=colors_map.get(status, "#ccc"))
                bottom += vals
        ax.set_ylabel("Jumlah SKU", fontsize=9)
        ax.set_title("Distribusi Status Stok per Kota", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Narasi per kota: kota dengan understock tertinggi
        if "Understock" in city_stat.columns:
            worst_city = city_stat.loc[city_stat["Understock"].idxmax(), "City"]
            worst_n    = city_stat["Understock"].max()
            st.caption(f"⚠️ Kota dengan understock terbanyak: **{worst_city}** ({worst_n} SKU). "
                       f"Segera prioritaskan pengiriman ke kota ini.")

    # ── 3. Insight LGBM: Produk yang paling banyak dikoreksi ─────────────────
    if has_lgbm and not df_elig.empty and "LGBM_Koreksi" in df_elig.columns:
        st.markdown("#### 🤖 Insight Koreksi LGBM")

        df_elig_nz = df_elig[df_elig["LGBM_Koreksi"].abs() > 0.5]
        n_up   = (df_elig["LGBM_Koreksi"] > 0.5).sum()
        n_down = (df_elig["LGBM_Koreksi"] < -0.5).sum()
        n_same = len(df_elig) - n_up - n_down

        st.markdown(f"""
        Dari **{len(df_elig):,} SKU** yang dikoreksi LGBM:
        - 📈 **{n_up:,} SKU** dinaikkan prediksinya (WMA cenderung *underestimate*)
        - 📉 **{n_down:,} SKU** diturunkan prediksinya (WMA cenderung *overestimate*)
        - ➡️ **{n_same:,} SKU** tidak berubah signifikan (koreksi < 0.5 unit)

        _Koreksi ke atas umumnya terjadi pada produk dengan permintaan yang sedang naik
        namun belum tercermin penuh dalam 3 bulan WMA. Koreksi ke bawah menunjukkan
        WMA masih "teringat" periode permintaan tinggi yang sudah berlalu._
        """)

        # Top 10 produk dengan koreksi terbesar
        if not df_elig_nz.empty:
            top_corr = (df_elig_nz.groupby(["No. Barang", "Nama Barang"])["LGBM_Koreksi"]
                        .mean().abs().nlargest(10).reset_index())
            top_corr.columns = ["No. Barang", "Nama Barang", "Koreksi Rata-rata (abs)"]
            with st.expander("🔝 Top 10 Produk dengan Koreksi LGBM Terbesar"):
                st.dataframe(top_corr.style.format({"Koreksi Rata-rata (abs)": "{:.2f}"}),
                             use_container_width=True, hide_index=True)

    # ── 4. Insight Kategori ABC ───────────────────────────────────────────────
    if KAT_COL in df_w.columns:
        st.markdown("#### 🅰️ Distribusi Kategori ABC")
        abc_dist = df_w.groupby(KAT_COL).agg(
            Jumlah_SKU=("No. Barang", "nunique"),
            SO_WMA_Total=("SO WMA", "sum"),
        ).reset_index()

        if has_lgbm and "SO_Final" in df_w.columns:
            abc_dist2 = df_w.groupby(KAT_COL)["SO_Final"].sum().reset_index()
            abc_dist  = abc_dist.merge(abc_dist2, on=KAT_COL, how="left")
            abc_dist.rename(columns={"SO_Final": "SO_Final_Total"}, inplace=True)

        abc_dist.rename(columns={KAT_COL: "Kategori ABC"}, inplace=True)
        st.dataframe(abc_dist.style.format({
            "SO_WMA_Total": "{:,.0f}",
            "SO_Final_Total": "{:,.0f}",
        }), use_container_width=True, hide_index=True)

        top_abc = abc_dist.loc[abc_dist["SO_WMA_Total"].idxmax(), "Kategori ABC"]
        st.caption(f"💡 Kategori **{top_abc}** mendominasi volume SO. "
                   f"Pastikan stok kategori ini selalu tersedia karena berkontribusi paling besar.")
