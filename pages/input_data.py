"""
pages/input_data.py
Halaman Input Data — muat file dari Google Drive + upload manual dari laptop.
Data dari kedua sumber digabungkan otomatis.
"""

import streamlit as st
import pandas as pd

from utils import (
    list_files_in_folder,
    download_and_read,
    read_produk_file,
    read_stock_file,
    FOLDER_PENJUALAN,
    FOLDER_PRODUK,
    FOLDER_STOCK,
)


def _read_uploaded_file(uploaded_file) -> pd.DataFrame:
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return pd.DataFrame()


def _normalize_penjualan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "No. Barang" in df.columns:
        df["No. Barang"] = df["No. Barang"].astype(str).str.strip()
    if "Tgl Faktur" in df.columns:
        df["Tgl Faktur"] = pd.to_datetime(df["Tgl Faktur"], errors="coerce")
    return df


def render(drive_service):
    st.title("📥 Input Data")
    st.markdown(
        "Muat data dari **Google Drive** dan/atau **upload manual dari laptop**. "
        "Data dari kedua sumber akan **digabungkan otomatis**."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 1. DATA PENJUALAN / SO
    # ══════════════════════════════════════════════════════════════════════════
    st.header("1. Data Penjualan / SO")

    tab_drive, tab_manual = st.tabs(["☁️ Dari Google Drive", "💻 Upload dari Laptop"])

    with tab_drive:
        st.markdown("Semua file penjualan di Google Drive akan **digabungkan otomatis**.")
        with st.spinner("Mencari file di Google Drive..."):
            penjualan_files = list_files_in_folder(drive_service, FOLDER_PENJUALAN)

        if penjualan_files:
            st.info(f"Ditemukan **{len(penjualan_files)} file** di Google Drive.")
            with st.expander("Lihat daftar file"):
                for f in penjualan_files:
                    st.text(f"• {f['name']}")
        else:
            st.warning("⚠️ Tidak ada file di folder Google Drive.")

        if st.button("☁️ Muat / Muat Ulang dari Google Drive", key="btn_gdrive"):
            if penjualan_files:
                with st.spinner("Mengunduh dan menggabungkan..."):
                    dfs = [download_and_read(drive_service, f["id"], f["name"])
                           for f in penjualan_files]
                    dfs = [_normalize_penjualan(d) for d in dfs if not d.empty]
                if dfs:
                    st.session_state["_penj_drive"] = pd.concat(dfs, ignore_index=True)
                    st.success(f"✅ {len(dfs)} file dimuat "
                               f"({len(st.session_state['_penj_drive']):,} baris).")
                else:
                    st.error("Gagal memuat data.")
            else:
                st.warning("Tidak ada file di Google Drive.")

    with tab_manual:
        st.markdown("""
        Upload satu atau beberapa file penjualan dari laptop. Format kolom harus sama
        dengan file Google Drive: `No. Faktur`, `Tgl Faktur`, `No. Barang`,
        `Qty` / `Kuantitas`, `Dept.`, `Nama Pelanggan`.
        """)
        uploaded = st.file_uploader(
            "Pilih file penjualan (.xlsx / .csv) — bisa lebih dari satu",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=True,
            key="uf_penjualan",
        )
        if uploaded:
            parts = []
            for uf in uploaded:
                df_tmp = _normalize_penjualan(_read_uploaded_file(uf))
                if not df_tmp.empty:
                    parts.append(df_tmp)
                    st.success(f"✅ `{uf.name}` — {len(df_tmp):,} baris")
                else:
                    st.warning(f"⚠️ `{uf.name}` kosong atau gagal dibaca.")
            if parts:
                st.session_state["_penj_manual"] = pd.concat(parts, ignore_index=True)

    # ── Gabungkan & tampilkan status ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔗 Status Penggabungan Data Penjualan")

    df_drive  = st.session_state.get("_penj_drive",  pd.DataFrame())
    df_manual = st.session_state.get("_penj_manual", pd.DataFrame())

    c1, c2, c3 = st.columns(3)
    c1.metric("Google Drive",   f"{len(df_drive):,} baris"  if not df_drive.empty  else "Belum dimuat")
    c2.metric("Upload Laptop",  f"{len(df_manual):,} baris" if not df_manual.empty else "Belum ada")

    parts = [d for d in [df_drive, df_manual] if not d.empty]
    if parts:
        df_gabung = pd.concat(parts, ignore_index=True)
        # Deduplikasi
        if "No. Faktur" in df_gabung.columns and "No. Barang" in df_gabung.columns:
            df_gabung["No. Faktur"]      = df_gabung["No. Faktur"].astype(str).str.strip()
            df_gabung["Faktur + Barang"] = df_gabung["No. Faktur"] + df_gabung["No. Barang"].astype(str)
            n_before = len(df_gabung)
            df_gabung.drop_duplicates(subset=["Faktur + Barang"], keep="first", inplace=True)
            n_dup = n_before - len(df_gabung)
            if n_dup:
                st.info(f"ℹ️ {n_dup:,} baris duplikat dihapus saat penggabungan.")
        c3.metric("Total Gabungan", f"{len(df_gabung):,} baris")
        st.session_state.df_penjualan = df_gabung

        if "Tgl Faktur" in df_gabung.columns:
            tmin = df_gabung["Tgl Faktur"].min()
            tmax = df_gabung["Tgl Faktur"].max()
            st.success(f"✅ Data aktif: **{tmin.date()}** s/d **{tmax.date()}** "
                       f"| **{len(df_gabung):,} transaksi**")

        with st.expander("Preview 20 baris pertama"):
            st.dataframe(df_gabung.head(20))

    else:
        c3.metric("Total Gabungan", "0 baris")
        st.warning("⚠️ Belum ada data penjualan. Muat dari Google Drive atau upload dari laptop.")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. PRODUK REFERENSI
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("2. Produk Referensi")

    tab_pd, tab_pm = st.tabs(["☁️ Dari Google Drive", "💻 Upload dari Laptop"])

    with tab_pd:
        with st.spinner("Mencari file produk..."):
            produk_files = list_files_in_folder(drive_service, FOLDER_PRODUK)
        sel_produk = st.selectbox(
            "Pilih file Produk:",
            options=[None] + produk_files,
            format_func=lambda x: x["name"] if x else "— Pilih file —",
            key="sel_produk",
        )
        if sel_produk:
            with st.spinner(f"Memuat {sel_produk['name']}..."):
                df_tmp = read_produk_file(drive_service, sel_produk["id"])
            if not df_tmp.empty:
                st.session_state.produk_ref = df_tmp
                st.success(f"✅ {len(df_tmp):,} produk dimuat dari Drive.")
            else:
                st.error("Gagal memuat file produk.")

    with tab_pm:
        st.markdown("Kolom wajib: `No. Barang`, `BRAND Barang`, `Kategori Barang`, `Nama Barang`")
        uf_produk = st.file_uploader("File produk referensi (.xlsx / .csv)", type=["xlsx", "xls", "csv"], key="uf_produk")
        if uf_produk:
            df_pm = _read_uploaded_file(uf_produk)
            df_pm.rename(columns={"Keterangan Barang": "Nama Barang"}, inplace=True, errors="ignore")
            required = ["No. Barang", "BRAND Barang", "Kategori Barang", "Nama Barang"]
            missing  = [c for c in required if c not in df_pm.columns]
            if missing:
                st.error(f"Kolom tidak ditemukan: {missing}")
            elif not df_pm.empty:
                st.session_state.produk_ref = df_pm[required].dropna(subset=["No. Barang"])
                st.success(f"✅ {len(st.session_state.produk_ref):,} produk dimuat dari laptop.")

    if not st.session_state.get("produk_ref", pd.DataFrame()).empty:
        with st.expander("Preview Produk Referensi"):
            st.dataframe(st.session_state.produk_ref.head(10))

    # ══════════════════════════════════════════════════════════════════════════
    # 3. DATA STOCK
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("3. Data Stock")

    tab_sd, tab_sm = st.tabs(["☁️ Dari Google Drive", "💻 Upload dari Laptop"])

    with tab_sd:
        with st.spinner("Mencari file stock..."):
            stock_files = list_files_in_folder(drive_service, FOLDER_STOCK)
        sel_stock = st.selectbox(
            "Pilih file Stock:",
            options=[None] + stock_files,
            format_func=lambda x: x["name"] if x else "— Pilih file —",
            key="sel_stock",
        )
        if sel_stock:
            with st.spinner(f"Memuat {sel_stock['name']}..."):
                df_tmp = read_stock_file(drive_service, sel_stock["id"])
            if not df_tmp.empty:
                st.session_state.df_stock      = df_tmp
                st.session_state.stock_filename = sel_stock["name"]
                st.success(f"✅ '{sel_stock['name']}' berhasil dimuat.")
            else:
                st.error("Gagal memuat file stock.")

    with tab_sm:
        st.markdown("""
        Upload file stock dari laptop. Format harus identik dengan file di Drive
        (skiprows=9, kolom gudang: `A - ITC`, `B`, `C`, dst.).
        """)
        uf_stock = st.file_uploader("File stock (.xlsx)", type=["xlsx", "xls"], key="uf_stock")
        if uf_stock:
            try:
                df_sm = pd.read_excel(uf_stock, sheet_name="Sheet1", skiprows=9, header=None)
                header = [
                    "No. Barang", "Keterangan Barang",
                    "A - ITC", "AT - TRANSIT ITC", "B", "BT - TRANSIT JKT",
                    "C", "C6", "CT - TRANSIT PUSAT", "D - SMG", "DT - TRANSIT SMG",
                    "E - JOG", "ET - TRANSIT JOG", "F - MLG", "FT - TRANSIT MLG",
                    "H - BALI", "HT - TRANSIT BALI", "X", "Y - SBY", "Y3 - Display Y", "YT - TRANSIT Y",
                ]
                df_sm.columns = header[:len(df_sm.columns)]
                st.session_state.df_stock      = df_sm
                st.session_state.stock_filename = uf_stock.name
                st.success(f"✅ '{uf_stock.name}' berhasil dimuat — {len(df_sm):,} SKU.")
            except Exception as e:
                st.error(f"Gagal memuat: {e}")

    if not st.session_state.get("df_stock", pd.DataFrame()).empty:
        with st.expander("Preview Data Stock"):
            st.dataframe(st.session_state.df_stock.head(5))

    # ══════════════════════════════════════════════════════════════════════════
    # RINGKASAN STATUS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📋 Ringkasan Status Data")

    df_p = st.session_state.get("df_penjualan", pd.DataFrame())
    df_r = st.session_state.get("produk_ref",   pd.DataFrame())
    df_s = st.session_state.get("df_stock",     pd.DataFrame())

    cols = st.columns(3)
    items = [("Data Penjualan", df_p, f"{len(df_p):,} transaksi"),
             ("Produk Referensi", df_r, f"{len(df_r):,} produk"),
             ("Data Stock", df_s, f"{len(df_s):,} SKU")]

    for i, (label, df, detail) in enumerate(items):
        if not df.empty:
            cols[i].success(f"✅ **{label}**\n\n{detail}")
        else:
            cols[i].error(f"❌ **{label}**\n\nBelum dimuat")

    if all(not df.empty for _, df, _ in items):
        st.success("🎉 Semua data sudah siap! Lanjut ke halaman analisis.")
    else:
        st.info("ℹ️ Lengkapi semua data di atas sebelum menjalankan analisis.")
