"""
app.py  ←  Entry point utama aplikasi Streamlit.
"""

import streamlit as st

st.set_page_config(layout="wide", page_title="Analisis Stock & ABC")

st.markdown("""
    <style>
        [data-testid="stSidebarNav"] { display: none; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.image(
    "https://eq-cdn.equiti-me.com/website/images/What_does_a_stock_split_mean.2e16d0ba.fill-1600x900.jpg",
    use_container_width=True,
)
st.sidebar.title("Analisis Stock dan ABC")

page = st.sidebar.radio(
    "Menu Navigasi:",
    (
        "Input Data",
        "Hasil Analisa Stock",
        "Hasil Analisa Stock V2",
        "Analisis Donor Stock",
        "Analisis Produk Baru",
        "🤖 Analisa Stock + LGBM",
    ),
    help="Pilih halaman untuk ditampilkan.",
)
st.sidebar.markdown("---")

import pandas as pd
_defaults = {
    "df_penjualan":          pd.DataFrame(),
    "produk_ref":            pd.DataFrame(),
    "df_stock":              pd.DataFrame(),
    "stock_filename":        "",
    "stock_analysis_result": None,
    "abc_analysis_result":   None,
    "bulan_columns_stock":   [],
    "df_portal_analyzed":    pd.DataFrame(),
    "stock_pivot_df":        pd.DataFrame(),
    # V2
    "stock_v2_result":       None,
    "stock_v2_bulan_cols":   [],
    "stock_v2_pivot_df":     pd.DataFrame(),
    # Produk Baru
    "items_df":              pd.DataFrame(),
    "new_product_result":    None,
    # LGBM
    "lgbm_train_result":     None,
    "lgbm_train_dataset":    None,
    "lgbm_predict_result":   None,
    "lgbm_model_sess":       None,
    "lgbm_enc_sess":         None,
    "lgbm_v2_base":          None,
    # Input tambahan (drive + manual)
    "_penj_drive":           pd.DataFrame(),
    "_penj_manual":          pd.DataFrame(),
}
for key, default in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

from utils.gdrive import init_drive_service
drive_service, DRIVE_AVAILABLE = init_drive_service()

if not DRIVE_AVAILABLE and page != "Input Data":
    st.warning("Koneksi Google Drive tidak tersedia. Harap periksa kredensial.")

if page == "Input Data":
    if not DRIVE_AVAILABLE:
        st.warning("Tidak dapat melanjutkan karena koneksi ke Google Drive gagal.")
        st.stop()
    from pages.input_data import render
    render(drive_service)

elif page == "Hasil Analisa Stock":
    from pages.stock_analysis import render
    render()

elif page == "Hasil Analisa Stock V2":
    from pages.stock_analysis_v2 import render
    render()

elif page == "Analisis Donor Stock":
    from pages.stock_donor import render
    render()

elif page == "Analisis Produk Baru":
    from pages.new_product_analysis import render
    render()

elif page == "🤖 Analisa Stock + LGBM":
    from pages.lgbm_analysis import render
    render()
