"""
utils/gdrive.py
Semua fungsi koneksi dan operasi Google Drive.
"""

import os
import time
import streamlit as st
import pandas as pd
from io import BytesIO
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── Konstanta Folder ID ────────────────────────────────────────────────────────
FOLDER_PENJUALAN      = "1Okgw8qHVM8HyBwnTUFHbmYkNKqCcswNZ"
FOLDER_PRODUK         = "1UdGbFzZ2Wv83YZLNwdU-rgY-LXlczsFv"
FOLDER_STOCK          = "1PMeH_wvgRUnyiZyZ_wrmKAATX9JyWzq_"
FOLDER_HASIL_ANALISIS = "1TE4a8IegbWDKoVeLPG_oCbuU-qnhd1jE"
FOLDER_PORTAL         = "1GOKVWugUMqN9aOWYCeFlKj-qTr2dA7_u"

SCOPES = ["https://www.googleapis.com/auth/drive"]


# ── Inisialisasi Koneksi ───────────────────────────────────────────────────────
def init_drive_service():
    """
    Inisialisasi Google Drive service dari secrets atau file credentials.json.
    Mengembalikan (drive_service, True) jika berhasil, (None, False) jika gagal.
    """
    try:
        if "gcp_service_account" in st.secrets:
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"], scopes=SCOPES
            )
            st.sidebar.success("Terhubung ke Google Drive.", icon="☁️")
        elif os.path.exists("credentials.json"):
            credentials = service_account.Credentials.from_service_account_file(
                "credentials.json", scopes=SCOPES
            )
            st.sidebar.success("Terhubung ke Google Drive.", icon="💻")
        else:
            st.sidebar.error("Kredensial Google Drive tidak ditemukan.")
            return None, False

        service = build("drive", "v3", credentials=credentials)
        return service, True

    except Exception as e:
        st.sidebar.error("Gagal terhubung ke Google Drive.")
        st.error(f"Detail Error: {e}")
        return None, False


# ── Helper: Exponential Backoff ────────────────────────────────────────────────
def _with_backoff(fn, retries: int = 5):
    """Jalankan fn() dengan exponential backoff jika terjadi error."""
    for i in range(retries):
        try:
            return fn()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


# ── List & Download ────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def list_files_in_folder(_drive_service, folder_id: str) -> list:
    """List semua file (bukan folder) di dalam folder Google Drive."""
    try:
        def _list():
            query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder'"
            resp = _drive_service.files().list(q=query, fields="files(id, name)").execute()
            return resp.get("files", [])
        return _with_backoff(_list)
    except Exception:
        return []


@st.cache_data(ttl=600)
def download_file_from_gdrive(_drive_service, file_id: str) -> BytesIO | None:
    """
    Download file dari Google Drive ke BytesIO.
    FIX: drive_service sekarang dioper sebagai parameter eksplisit
         (tidak lagi bergantung pada global scope) agar cache aman.
    """
    try:
        def _download():
            request = _drive_service.files().get_media(fileId=file_id)
            fh = BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            fh.seek(0)
            return fh
        return _with_backoff(_download)
    except Exception as e:
        st.error(f"Gagal mengunduh file {file_id}. Error: {e}")
        return None


# ── Read Helper ────────────────────────────────────────────────────────────────
def download_and_read(_drive_service, file_id: str, file_name: str, **kwargs) -> pd.DataFrame:
    """Download lalu baca sebagai DataFrame (CSV atau Excel)."""
    fh = download_file_from_gdrive(_drive_service, file_id)
    if fh is None:
        return pd.DataFrame()
    if file_name.endswith(".csv"):
        return pd.read_csv(fh, **kwargs)
    return pd.read_excel(fh, **kwargs)


def read_produk_file(_drive_service, file_id: str) -> pd.DataFrame:
    """Baca file produk referensi dari Google Drive."""
    fh = download_file_from_gdrive(_drive_service, file_id)
    if fh is None:
        return pd.DataFrame()
    df = pd.read_excel(fh, sheet_name="Sheet1 (2)", skiprows=6, usecols=[0, 1, 2, 3])
    df.columns = ["No. Barang", "BRAND Barang", "Kategori Barang", "Nama Barang"]
    return df


def read_stock_file(_drive_service, file_id: str) -> pd.DataFrame:
    """Baca file stock dari Google Drive."""
    fh = download_file_from_gdrive(_drive_service, file_id)
    if fh is None:
        return pd.DataFrame()
    df = pd.read_excel(fh, sheet_name="Sheet1", skiprows=9, header=None)
    header = [
        "No. Barang", "Keterangan Barang",
        "A - ITC", "AT - TRANSIT ITC", "B", "BT - TRANSIT JKT",
        "C", "C6", "CT - TRANSIT PUSAT", "D - SMG", "DT - TRANSIT SMG",
        "E - JOG", "ET - TRANSIT JOG", "F - MLG", "FT - TRANSIT MLG",
        "H - BALI", "HT - TRANSIT BALI", "X", "Y - SBY", "Y3 - Display Y", "YT - TRANSIT Y",
    ]
    df.columns = header[: len(df.columns)]
    return df
