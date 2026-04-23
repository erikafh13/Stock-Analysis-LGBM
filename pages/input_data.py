"""
pages/input_data.py
Halaman Input Data — muat file dari Google Drive ke session state.
"""

import streamlit as st
from utils import (
    list_files_in_folder,
    download_and_read,
    read_produk_file,
    read_stock_file,
    convert_df_to_excel,
    FOLDER_PENJUALAN,
    FOLDER_PRODUK,
    FOLDER_STOCK,
)


def render(drive_service):
    st.title("📥 Input Data")
    st.markdown("Muat atau muat ulang data yang diperlukan dari Google Drive.")

    # ── 1. Data Penjualan ──────────────────────────────────────────────────────
    st.header("1. Data Penjualan")
    with st.spinner("Mencari file penjualan di Google Drive..."):
        penjualan_files = list_files_in_folder(drive_service, FOLDER_PENJUALAN)

    if st.button("Muat / Muat Ulang Data Penjualan"):
        if penjualan_files:
            with st.spinner("Menggabungkan semua file penjualan..."):
                dfs = [
                    download_and_read(drive_service, f["id"], f["name"])
                    for f in penjualan_files
                ]
                dfs = [d for d in dfs if not d.empty]
                if dfs:
                    import pandas as pd
                    st.session_state.df_penjualan = pd.concat(dfs, ignore_index=True)
                    st.success("Data penjualan berhasil dimuat ulang.")
                else:
                    st.error("Gagal memuat data penjualan. Periksa koneksi atau file.")
        else:
            st.warning("⚠️ Tidak ada file penjualan ditemukan di folder Google Drive.")

    if not st.session_state.df_penjualan.empty:
        st.success("✅ Data penjualan telah dimuat.")
        st.dataframe(st.session_state.df_penjualan)
        st.download_button(
            label="📥 Unduh Data Penjualan Gabungan (Excel)",
            data=convert_df_to_excel(st.session_state.df_penjualan),
            file_name="data_penjualan_gabungan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # ── 2. Produk Referensi ────────────────────────────────────────────────────
    st.header("2. Produk Referensi")
    with st.spinner("Mencari file produk di Google Drive..."):
        produk_files = list_files_in_folder(drive_service, FOLDER_PRODUK)

    selected_produk = st.selectbox(
        "Pilih file Produk dari Google Drive (pilih 1 file):",
        options=[None] + produk_files,
        format_func=lambda x: x["name"] if x else "Pilih file",
    )
    if selected_produk:
        with st.spinner(f"Memuat file {selected_produk['name']}..."):
            df_temp = read_produk_file(drive_service, selected_produk["id"])
            if not df_temp.empty:
                st.session_state.produk_ref = df_temp
                st.success(f"File '{selected_produk['name']}' berhasil dimuat.")
            else:
                st.error(f"Gagal memuat file '{selected_produk['name']}'.")

    if not st.session_state.produk_ref.empty:
        st.dataframe(st.session_state.produk_ref.head())

    # ── 3. Data Stock ──────────────────────────────────────────────────────────
    st.header("3. Data Stock")
    with st.spinner("Mencari file stock di Google Drive..."):
        stock_files = list_files_in_folder(drive_service, FOLDER_STOCK)

    selected_stock = st.selectbox(
        "Pilih file Stock dari Google Drive (pilih 1 file):",
        options=[None] + stock_files,
        format_func=lambda x: x["name"] if x else "Pilih file",
    )
    if selected_stock:
        with st.spinner(f"Memuat file {selected_stock['name']}..."):
            df_temp = read_stock_file(drive_service, selected_stock["id"])
            if not df_temp.empty:
                st.session_state.df_stock = df_temp
                st.session_state.stock_filename = selected_stock["name"]
                st.success(f"File stock '{selected_stock['name']}' berhasil dimuat.")
            else:
                st.error(f"Gagal memuat file stock '{selected_stock['name']}'.")

    if not st.session_state.df_stock.empty:
        st.dataframe(st.session_state.df_stock.head())
