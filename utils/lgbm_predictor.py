"""
utils/lgbm_predictor.py
─────────────────────────────────────────────────────────────────────────────
Modul Residual Learning: WMA + LightGBM Error Correction

Arsitektur:
    SO_Final = SO_WMA + LGBM_correction(residual)

Rolling Window:
    - window_end  = titik acuan (hari terakhir input WMA)
    - WMA input   = window_end - 89 hari → window_end      (90 hari)
    - Actual SO   = window_end + 1 hari  → window_end + 30 (30 hari)
    - Residual    = Actual_SO - SO_WMA
    - Step        = 30 hari (bukan by bulan kalender)

Fitur LGBM (disepakati):
    Dari WMA     : so_wma, penjualan_bln1, penjualan_bln2, penjualan_bln3
    Dari Kategori: kategori_abc_enc, kategori_brg_enc, brand_enc
    Dari Temporal: hari_ke (posisi dalam timeline), bulan (1-12)
    Dari Kota    : city_enc
    Dari History : residual_lag1, residual_lag2

Keputusan desain:
    - Kategori F & E di-exclude (pakai WMA saja)
    - Satu model global untuk semua produk & kota
    - Model disimpan ke Google Drive
    - Train/test split kronologis (bukan random)
─────────────────────────────────────────────────────────────────────────────
"""

import math
import warnings
import joblib
import numpy as np
import pandas as pd
from io import BytesIO

warnings.filterwarnings("ignore")

# ── Konstanta ──────────────────────────────────────────────────────────────────
MODEL_FILENAME   = "lgbm_residual_model.joblib"
ENCODER_FILENAME = "lgbm_label_encoders.joblib"

EXCLUDE_CATEGORIES = {"E", "F"}

WMA_W1, WMA_W2, WMA_W3 = 0.5, 0.3, 0.2

FEATURE_COLS = [
    "so_wma",
    "penjualan_bln1",
    "penjualan_bln2",
    "penjualan_bln3",
    "kategori_abc_enc",
    "kategori_brg_enc",
    "brand_enc",
    "hari_ke",
    "bulan",
    "city_enc",
    "residual_lag1",
    "residual_lag2",
]

TARGET_COL = "residual"


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _wma_single(s1: float, s2: float, s3: float) -> float:
    return math.ceil((s1 * WMA_W1) + (s2 * WMA_W2) + (s3 * WMA_W3))


def _hitung_s1s2s3(df_so: pd.DataFrame, window_end: pd.Timestamp) -> pd.DataFrame:
    """Hitung s1, s2, s3 per No.Barang × City untuk satu window_end."""
    r1_start = window_end - pd.DateOffset(days=29)
    r2_start = window_end - pd.DateOffset(days=59)
    r2_end   = window_end - pd.DateOffset(days=30)
    r3_start = window_end - pd.DateOffset(days=89)
    r3_end   = window_end - pd.DateOffset(days=60)

    mask = df_so["Tgl Faktur"].between(r3_start, window_end)
    df_w = df_so[mask]
    if df_w.empty:
        return pd.DataFrame(columns=["No. Barang", "City", "s1", "s2", "s3"])

    s1 = (df_w[df_w["Tgl Faktur"].between(r1_start, window_end)]
          .groupby(["No. Barang", "City"])["Kuantitas"].sum().reset_index(name="s1"))
    s2 = (df_w[df_w["Tgl Faktur"].between(r2_start, r2_end)]
          .groupby(["No. Barang", "City"])["Kuantitas"].sum().reset_index(name="s2"))
    s3 = (df_w[df_w["Tgl Faktur"].between(r3_start, r3_end)]
          .groupby(["No. Barang", "City"])["Kuantitas"].sum().reset_index(name="s3"))

    return (s1.merge(s2, on=["No. Barang", "City"], how="outer")
              .merge(s3, on=["No. Barang", "City"], how="outer")
              .fillna(0))


# ══════════════════════════════════════════════════════════════════════════════
# 1. BUILDER: Rolling Window Dataset
# ══════════════════════════════════════════════════════════════════════════════

def build_training_dataset(
    df_so: pd.DataFrame,
    df_abc: pd.DataFrame,
    step_days: int = 30,
) -> pd.DataFrame:
    """
    Bangun dataset training menggunakan rolling window per step_days hari.

    Setiap window menghasilkan satu baris per No.Barang × City:
        Fitur  : s1, s2, s3, so_wma, kategori, kota, temporal, lag residual
        Target : residual = actual_SO (30 hari ke depan) - so_wma
    """
    df_so = df_so.copy()
    df_so["Tgl Faktur"] = pd.to_datetime(df_so["Tgl Faktur"])
    df_so["City"]       = df_so["City"].str.upper().str.strip()
    df_so["Kuantitas"]  = pd.to_numeric(df_so["Kuantitas"], errors="coerce").fillna(0)

    start_global = df_so["Tgl Faktur"].min()
    end_global   = df_so["Tgl Faktur"].max()

    first_end = start_global + pd.DateOffset(days=89)
    last_end  = end_global   - pd.DateOffset(days=30)

    if first_end > last_end:
        return pd.DataFrame()

    # Referensi ABC
    df_abc_ref = df_abc[[
        "No. Barang", "City",
        "Kategori ABC (Log-Benchmark - WMA)",
        "Kategori Barang", "BRAND Barang",
    ]].copy()
    df_abc_ref["City"] = df_abc_ref["City"].str.upper().str.strip()
    df_abc_ref = df_abc_ref.rename(columns={
        "Kategori ABC (Log-Benchmark - WMA)": "kategori_abc",
        "BRAND Barang": "brand",
    }).drop_duplicates(subset=["No. Barang", "City"])

    records    = []
    window_end = first_end

    while window_end <= last_end:
        actual_start = window_end + pd.DateOffset(days=1)
        actual_end   = window_end + pd.DateOffset(days=30)

        s_df = _hitung_s1s2s3(df_so, window_end)
        if s_df.empty:
            window_end += pd.DateOffset(days=step_days)
            continue

        mask_actual = df_so["Tgl Faktur"].between(actual_start, actual_end)
        actual_df   = (df_so[mask_actual]
                       .groupby(["No. Barang", "City"])["Kuantitas"]
                       .sum().reset_index(name="actual_so"))

        window_df = s_df.merge(actual_df, on=["No. Barang", "City"], how="left")
        window_df["actual_so"] = window_df["actual_so"].fillna(0)
        window_df["so_wma"]    = window_df.apply(
            lambda r: float(_wma_single(r["s1"], r["s2"], r["s3"])), axis=1)
        window_df["residual"]       = window_df["actual_so"] - window_df["so_wma"]
        window_df["penjualan_bln1"] = window_df["s1"].astype(float)
        window_df["penjualan_bln2"] = window_df["s2"].astype(float)
        window_df["penjualan_bln3"] = window_df["s3"].astype(float)
        window_df["window_end"]     = window_end
        window_df["hari_ke"]        = (window_end - start_global).days
        window_df["bulan"]          = window_end.month

        records.append(window_df[[
            "No. Barang", "City", "window_end", "hari_ke", "bulan",
            "so_wma", "penjualan_bln1", "penjualan_bln2", "penjualan_bln3",
            "actual_so", TARGET_COL,
        ]])

        window_end += pd.DateOffset(days=step_days)

    if not records:
        return pd.DataFrame()

    df_train = pd.concat(records, ignore_index=True)
    df_train = df_train.merge(df_abc_ref, on=["No. Barang", "City"], how="left")
    df_train["kategori_abc"]    = df_train["kategori_abc"].fillna("E")
    df_train["Kategori Barang"] = df_train["Kategori Barang"].fillna("UNKNOWN")
    df_train["brand"]           = df_train["brand"].fillna("UNKNOWN")

    df_train = df_train[~df_train["kategori_abc"].isin(EXCLUDE_CATEGORIES)].copy()

    # Filter produk yang terlalu jarang (minimal 3 window)
    wcount = (df_train.groupby(["No. Barang", "City"])["window_end"]
              .nunique().reset_index(name="n_window"))
    valid  = wcount[wcount["n_window"] >= 3][["No. Barang", "City"]]
    df_train = df_train.merge(valid, on=["No. Barang", "City"], how="inner")

    # Residual lag
    df_train = df_train.sort_values(["No. Barang", "City", "window_end"])
    grp = df_train.groupby(["No. Barang", "City"])[TARGET_COL]
    df_train["residual_lag1"] = grp.shift(1).fillna(0.0)
    df_train["residual_lag2"] = grp.shift(2).fillna(0.0)

    return df_train.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_lgbm_model(df_train_raw: pd.DataFrame, test_ratio: float = 0.3) -> dict:
    """Training LightGBM dari output build_training_dataset(). Split kronologis."""
    try:
        import lightgbm as lgb
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import mean_absolute_error, mean_squared_error
    except ImportError as e:
        raise ImportError(f"Library tidak tersedia: {e}. Tambahkan ke requirements.txt")

    df = df_train_raw.copy()

    encoders   = {}
    encode_map = [
        ("City",            "city_enc"),
        ("kategori_abc",    "kategori_abc_enc"),
        ("Kategori Barang", "kategori_brg_enc"),
        ("brand",           "brand_enc"),
    ]
    for col, feat_col in encode_map:
        le = LabelEncoder()
        df[feat_col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Simpan start_global untuk konsistensi predict
    min_window  = df["window_end"].min()
    min_hari_ke = df[df["window_end"] == min_window]["hari_ke"].iloc[0]
    encoders["_start_global"] = min_window - pd.DateOffset(days=int(min_hari_ke))

    df        = df.sort_values("window_end").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    df_tr     = df.iloc[:split_idx].copy()
    df_te     = df.iloc[split_idx:].copy()

    X_train, y_train = df_tr[FEATURE_COLS], df_tr[TARGET_COL]
    X_test,  y_test  = df_te[FEATURE_COLS], df_te[TARGET_COL]

    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])

    def _metrics(actual, wma, final, n):
        mae_w  = mean_absolute_error(actual, wma)
        rmse_w = mean_squared_error(actual, wma)  ** 0.5
        mae_f  = mean_absolute_error(actual, final)
        rmse_f = mean_squared_error(actual, final) ** 0.5
        imp    = (mae_w - mae_f) / mae_w * 100 if mae_w > 0 else 0
        return {"n_samples": n,
                "mae_wma": round(mae_w, 4), "rmse_wma": round(rmse_w, 4),
                "mae_final": round(mae_f, 4), "rmse_final": round(rmse_f, 4),
                "improvement_mae_pct": round(imp, 2)}

    pred_tr = model.predict(X_train)
    pred_te = model.predict(X_test)
    metrics = {
        "train": _metrics(df_tr["actual_so"], df_tr["so_wma"], df_tr["so_wma"] + pred_tr, len(df_tr)),
        "test":  _metrics(df_te["actual_so"], df_te["so_wma"], df_te["so_wma"] + pred_te, len(df_te)),
    }

    df_te = df_te.copy()
    df_te["lgbm_koreksi"] = pred_te
    df_te["so_final"]     = df_te["so_wma"] + pred_te

    feat_importance = pd.Series(
        model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

    return {
        "model": model, "encoders": encoders, "metrics": metrics,
        "df_test": df_te, "feature_importance": feat_importance,
        "n_windows": df["window_end"].nunique(),
        "train_cutoff": df.iloc[split_idx]["window_end"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. SAVE & LOAD MODEL (Google Drive)
# ══════════════════════════════════════════════════════════════════════════════

def save_model_to_gdrive(model, encoders, drive_service, folder_id) -> bool:
    try:
        from googleapiclient.http import MediaIoBaseUpload

        def _upload(obj, filename):
            buf = BytesIO()
            joblib.dump(obj, buf)
            buf.seek(0)
            query    = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
            existing = drive_service.files().list(q=query, fields="files(id)").execute().get("files", [])
            for f in existing:
                drive_service.files().delete(fileId=f["id"]).execute()
            media = MediaIoBaseUpload(buf, mimetype="application/octet-stream")
            drive_service.files().create(
                body={"name": filename, "parents": [folder_id]},
                media_body=media, fields="id").execute()

        _upload(model,    MODEL_FILENAME)
        _upload(encoders, ENCODER_FILENAME)
        return True
    except Exception as e:
        print(f"[lgbm_predictor] Gagal simpan: {e}")
        return False


def load_model_from_gdrive(drive_service, folder_id) -> tuple:
    try:
        from googleapiclient.http import MediaIoBaseDownload

        def _download(filename):
            query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
            files = drive_service.files().list(q=query, fields="files(id)").execute().get("files", [])
            if not files:
                return None
            buf  = BytesIO()
            req  = drive_service.files().get_media(fileId=files[0]["id"])
            dl   = MediaIoBaseDownload(buf, req)
            done = False
            while not done:
                _, done = dl.next_chunk()
            buf.seek(0)
            return joblib.load(buf)

        model    = _download(MODEL_FILENAME)
        encoders = _download(ENCODER_FILENAME)
        if model is None or encoders is None:
            return None, None
        return model, encoders
    except Exception as e:
        print(f"[lgbm_predictor] Gagal load: {e}")
        return None, None


def check_model_exists_in_gdrive(drive_service, folder_id) -> bool:
    try:
        query = f"name='{MODEL_FILENAME}' and '{folder_id}' in parents and trashed=false"
        files = drive_service.files().list(q=query, fields="files(id)").execute().get("files", [])
        return len(files) > 0
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 4. PREDICT
# ══════════════════════════════════════════════════════════════════════════════

def predict_correction(df_current, model, encoders, df_so_raw, end_date) -> pd.DataFrame:
    """
    Prediksi koreksi LGBM untuk data analisis terkini.
    Fitur dibangun ulang dari df_so_raw agar konsisten dengan training.
    """
    end_dt       = pd.to_datetime(end_date)
    start_global = pd.to_datetime(encoders.get("_start_global", end_dt))

    df = df_current.copy()
    df["City_upper"] = df["City"].str.upper().str.strip()
    df["LGBM_Koreksi"]  = 0.0
    df["SO_Final"]       = df["SO WMA"].apply(math.ceil).astype(float)
    df["LGBM_Available"] = False

    kat_col       = "Kategori ABC (Log-Benchmark - WMA)"
    eligible_mask = ~df[kat_col].isin(EXCLUDE_CATEGORIES)

    if eligible_mask.sum() == 0:
        return df

    df_so = df_so_raw.copy()
    df_so["Tgl Faktur"] = pd.to_datetime(df_so["Tgl Faktur"])
    df_so["City"]       = df_so["City"].str.upper().str.strip()
    df_so["Kuantitas"]  = pd.to_numeric(df_so["Kuantitas"], errors="coerce").fillna(0)

    s_df   = _hitung_s1s2s3(df_so, end_dt)
    df_elig = df[eligible_mask].copy()

    df_elig = df_elig.merge(
        s_df.rename(columns={"s1": "penjualan_bln1", "s2": "penjualan_bln2", "s3": "penjualan_bln3"}),
        left_on=["No. Barang", "City_upper"], right_on=["No. Barang", "City"],
        how="left", suffixes=("", "_s"))
    for col in ["penjualan_bln1", "penjualan_bln2", "penjualan_bln3"]:
        df_elig[col] = df_elig[col].fillna(0.0)

    df_elig["hari_ke"] = (end_dt - start_global).days
    df_elig["bulan"]   = end_dt.month
    df_elig["so_wma"]  = df_elig["SO WMA"].astype(float)

    def _safe_encode(le, values):
        known = set(le.classes_)
        return np.array([le.transform([v])[0] if v in known else 0 for v in values])

    df_elig["city_enc"]         = _safe_encode(encoders["City"], df_elig["City_upper"].values)
    df_elig["kategori_abc_enc"] = _safe_encode(encoders["kategori_abc"], df_elig[kat_col].values)
    df_elig["kategori_brg_enc"] = _safe_encode(encoders["Kategori Barang"], df_elig["Kategori Barang"].values)
    df_elig["brand_enc"]        = _safe_encode(encoders["brand"], df_elig["BRAND Barang"].fillna("UNKNOWN").values)

    df_elig["residual_lag1"] = 0.0
    df_elig["residual_lag2"] = 0.0

    for lag_n, lag_days in [(1, 30), (2, 60)]:
        lag_end       = end_dt - pd.DateOffset(days=lag_days)
        lag_act_start = lag_end + pd.DateOffset(days=1)
        lag_act_end   = lag_end + pd.DateOffset(days=30)

        s_lag = _hitung_s1s2s3(df_so, lag_end)
        if s_lag.empty:
            continue

        mask_act = df_so["Tgl Faktur"].between(lag_act_start, lag_act_end)
        act_lag  = (df_so[mask_act].groupby(["No. Barang", "City"])["Kuantitas"]
                    .sum().reset_index(name="actual_lag"))

        lag_df = s_lag.merge(act_lag, on=["No. Barang", "City"], how="left")
        lag_df["actual_lag"] = lag_df["actual_lag"].fillna(0)
        lag_df["wma_lag"]    = lag_df.apply(
            lambda r: float(_wma_single(r["s1"], r["s2"], r["s3"])), axis=1)
        lag_df[f"res_lag{lag_n}"] = lag_df["actual_lag"] - lag_df["wma_lag"]

        df_elig = df_elig.merge(
            lag_df[["No. Barang", "City", f"res_lag{lag_n}"]],
            left_on=["No. Barang", "City_upper"], right_on=["No. Barang", "City"],
            how="left", suffixes=("", f"_lag{lag_n}"))
        df_elig[f"residual_lag{lag_n}"] = df_elig[f"res_lag{lag_n}"].fillna(0.0)

    koreksi  = model.predict(df_elig[FEATURE_COLS])
    so_final = np.ceil(df_elig["so_wma"].values + koreksi).clip(min=0)

    df.loc[eligible_mask, "LGBM_Koreksi"]  = np.round(koreksi, 2)
    df.loc[eligible_mask, "SO_Final"]       = so_final.astype(int)
    df.loc[eligible_mask, "LGBM_Available"] = True
    df["SO_Final"] = df["SO_Final"].astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUASI
# ══════════════════════════════════════════════════════════════════════════════

def summarize_metrics(metrics: dict) -> pd.DataFrame:
    rows = []
    for split, m in metrics.items():
        rows.append({
            "Split": split.upper(), "N Sampel": m["n_samples"],
            "MAE — WMA": m["mae_wma"], "MAE — SO Final": m["mae_final"],
            "RMSE — WMA": m["rmse_wma"], "RMSE — SO Final": m["rmse_final"],
            "Improvement MAE (%)": m["improvement_mae_pct"],
        })
    return pd.DataFrame(rows)
