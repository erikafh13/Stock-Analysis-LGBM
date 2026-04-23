"""
pages/stock_donor.py  —  Analisis Donor Stock (V3 Lateral Transfer)
"""

import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO

from utils.analysis import (
    highlight_status_stock,
    highlight_kategori_abc_log,
)

ALL_CITIES = ["SURABAYA", "JAKARTA", "SEMARANG", "JOGJA", "MALANG", "BALI"]

DEFAULT_DISTANCE_PRIORITY = {
    "JAKARTA":  ["SURABAYA", "SEMARANG", "JOGJA", "MALANG", "BALI"],
    "SEMARANG": ["SURABAYA", "JOGJA", "MALANG", "JAKARTA", "BALI"],
    "JOGJA":    ["SURABAYA", "SEMARANG", "MALANG", "BALI", "JAKARTA"],
    "MALANG":   ["SURABAYA", "JOGJA", "SEMARANG", "BALI", "JAKARTA"],
    "BALI":     ["SURABAYA", "MALANG", "JOGJA", "SEMARANG", "JAKARTA"],
    "SURABAYA": [],
}

_SKENARIO_COLOR = {
    "0 - TIDAK ADA KEBUTUHAN":      "#d4edda",
    "2 - SBY CUKUP":                "#cce5ff",
    "3 - SBY TERBATAS + ADA DONOR": "#fff3cd",
    "4 - HANYA DONOR CABANG":       "#ffe5b4",
    "5 - POOL TIDAK CUKUP":         "#f8d7da",
    "1 - KURANG (TIDAK ADA POOL)":  "#f5c6cb",
}

def _hl_ske(val):   return f"background-color: {_SKENARIO_COLOR.get(str(val), '')}"
def _hl_donor(val): return "background-color: #d4edda; font-weight: bold" if val and val != "-" else ""
def _hl_recv(val):  return "background-color: #cce5ff; font-weight: bold" if val and val != "-" else ""

def _default_rules():
    r = {}
    for d in ALL_CITIES:
        r[d] = {}
        for p in ALL_CITIES:
            r[d][p] = False if d == p else True
    return r


# ── Kalkulasi inti ─────────────────────────────────────────────────────────────
def _run_donor_calc(df, rules, distance):
    KAT_COL = "Kategori ABC (Log-Benchmark - WMA)"
    KEYS    = ["No. Barang", "Kategori Barang", "BRAND Barang", "Nama Barang"]
    all_rows = []

    for sku, group in df.groupby("No. Barang"):
        group = group.copy().reset_index(drop=True)
        mask_sby = group["City"] == "SURABAYA"

        sby_rows      = group[mask_sby]
        stock_sby     = float(sby_rows["Stock Cabang"].sum())
        min_stock_sby = float(sby_rows["Min Stock"].sum())
        sisa_sby      = max(0.0, stock_sby - min_stock_sby)

        cab_rows = group[~mask_sby].copy()

        donors = cab_rows[
            (cab_rows["Status Stock"] == "Overstock") &
            (cab_rows[KAT_COL] != "F")
        ].copy()
        donors["_avail"] = (donors["Stock Cabang"] - donors["Max Stock"]).clip(lower=0).astype(float)
        donors = donors[donors["_avail"] > 0]

        receivers = cab_rows[
            (cab_rows["Status Stock"] == "Understock") &
            (cab_rows[KAT_COL] != "F")
        ].copy().sort_values("Persentase Stock", ascending=True)

        total_need = float(receivers["Add Stock"].sum())
        donor_pool = float(donors["_avail"].sum())
        total_pool = sisa_sby + donor_pool

        if total_need == 0:
            skenario = "0 - TIDAK ADA KEBUTUHAN"
        elif total_pool == 0:
            skenario = "1 - KURANG (TIDAK ADA POOL)"
        elif sisa_sby >= total_need:
            skenario = "2 - SBY CUKUP"
        elif sisa_sby > 0 and len(donors) > 0:
            skenario = "3 - SBY TERBATAS + ADA DONOR"
        elif sisa_sby == 0 and len(donors) > 0:
            skenario = "4 - HANYA DONOR CABANG"
        else:
            skenario = "5 - POOL TIDAK CUKUP"

        donor_avail = {"SURABAYA": sisa_sby}
        for _, dr in donors.iterrows():
            donor_avail[dr["City"]] = dr["_avail"]

        qty_terima  = {}
        qty_donor   = {}
        terima_dari = {}
        donor_ke    = {}

        for _, rv in receivers.iterrows():
            rcity = rv["City"]
            need  = float(rv["Add Stock"])
            prio  = distance.get(rcity, [])
            extra = [c for c in donor_avail if c not in prio]
            for dcity in (prio + extra):
                if need <= 0:
                    break
                avail = donor_avail.get(dcity, 0.0)
                if avail <= 0:
                    continue
                if not rules.get(dcity, {}).get(rcity, True):
                    continue
                alloc = int(np.ceil(min(need, avail)))
                qty_terima[rcity]  = qty_terima.get(rcity, 0)  + alloc
                qty_donor[dcity]   = qty_donor.get(dcity, 0)   + alloc
                terima_dari.setdefault(rcity, []).append(dcity)
                donor_ke.setdefault(dcity, []).append(rcity)
                donor_avail[dcity] -= alloc
                need -= alloc

        first = group.iloc[0]
        meta  = {k: first.get(k, "") for k in KEYS}

        for _, row in group.iterrows():
            city = row["City"]
            bisa = int(max(0, row["Stock Cabang"] - row["Max Stock"])) if row["Status Stock"] == "Overstock" and row.get(KAT_COL) != "F" else 0
            sisa_po = int(max(0, row["Add Stock"] - qty_terima.get(city, 0))) if row["Status Stock"] == "Understock" else 0
            all_rows.append({
                **meta,
                "City":             city,
                "Kategori ABC":     row.get(KAT_COL, "-"),
                "Stock Cabang":     int(row["Stock Cabang"]),
                "Min Stock":        int(row["Min Stock"]),
                "Max Stock":        int(row["Max Stock"]),
                "Add Stock":        int(row["Add Stock"]),
                "Status Stock":     row["Status Stock"],
                "% Stock":          round(float(row.get("Persentase Stock", 0)), 1),
                "Qty_Bisa_Donor":   bisa,
                "Donor_Ke":         ", ".join(donor_ke.get(city, [])) or "-",
                "Qty_Donor":        int(qty_donor.get(city, 0)),
                "Terima_Dari":      ", ".join(terima_dari.get(city, [])) or "-",
                "Qty_Terima":       int(qty_terima.get(city, 0)),
                "Sisa_PO_Supplier": sisa_po,
                "Skenario":         skenario,
                "Total_Pool":       int(total_pool),
                "Total_Need":       int(total_need),
            })

    return pd.DataFrame(all_rows)


# ── Render ─────────────────────────────────────────────────────────────────────
def render():
    st.title("🔄 Analisis Donor Stock")
    st.caption("Distribusi stok lateral antar cabang — optimalkan sebelum PO ke supplier")

    result_v2 = st.session_state.get("stock_v2_result")
    if result_v2 is None or (isinstance(result_v2, pd.DataFrame) and result_v2.empty):
        st.warning("⚠️ Jalankan dulu **Hasil Analisa Stock V2**, kemudian kembali ke halaman ini.")
        st.stop()

    df = result_v2.copy()
    df = df[df["City"] != "OTHERS"]
    KAT_COL = "Kategori ABC (Log-Benchmark - WMA)"

    if "Persentase Stock" not in df.columns:
        df["Persentase Stock"] = np.where(
            df["Min Stock"] > 0,
            (df["Stock Cabang"] / df["Min Stock"]) * 100,
            np.where(df["Stock Cabang"] > 0, 10000, 0)
        ).round(1)

    # ── Penjelasan ─────────────────────────────────────────────────────────────
    with st.expander("📖 Cara Kerja & Arti Setiap Kolom (klik untuk membaca)", expanded=False):
        st.markdown("""
### Apa yang dihitung di sini?

Halaman ini menjawab: **"Sebelum order ke supplier, adakah cabang yang kelebihan stok
dan bisa kirim ke cabang yang kekurangan?"**

---

### Langkah Perhitungan (per SKU):

**① Siapa yang KELEBIHAN (Donor)?**
Cabang dengan status **Overstock** → stoknya melebihi Max Stock.
`Qty bisa didonorkan = Stock Cabang − Max Stock`
*Contoh: Jogja stock 75, Max 46 → bisa kirim 29 unit*

**② Siapa yang KEKURANGAN (Penerima)?**
Cabang dengan status **Understock** → stoknya di bawah Min Stock.
`Kebutuhan (Add Stock) = Min Stock − Stock Cabang`
*Contoh: Jakarta stock 6, Min 47 → butuh 41 unit*

**③ Berapa yang tersedia untuk dibagikan (Pool)?**
- Sisa Surabaya = `max(0, Stock SBY − Min Stock SBY)` → SBY hanya boleh kirim kalau stoknya masih di atas minimum sendiri
- Donor cabang = total kelebihan semua cabang overstock
- **Total Pool = Sisa SBY + semua donor cabang**

**④ Cara pembagian:**
- Penerima yang paling kritis (% stock terkecil) **mendapat prioritas pertama**
- Untuk tiap penerima, donor dipilih berdasarkan **urutan jarak** yang Anda atur
- Jika ada **aturan larangan**, donor tersebut dilewati
- Jika pool habis sebelum semua terpenuhi → sisanya masuk ke kolom **Sisa PO Supplier**

---

### Arti Setiap Kolom:

| Kolom | Artinya |
|---|---|
| **Stock Cabang** | Stok fisik saat ini |
| **Min Stock** | Batas bawah aman berdasarkan penjualan × buffer ABC |
| **Max Stock** | Batas atas ideal |
| **Add Stock** | Unit yang dibutuhkan agar mencapai Min Stock |
| **% Stock** | `(Stock ÷ Min Stock) × 100` — makin kecil makin kritis |
| **Qty_Bisa_Donor** | Unit kelebihan yang bisa didonorkan `(Stock − Max)` |
| **Donor_Ke** | Cabang mana yang menerima kiriman dari cabang ini 🟢 |
| **Qty_Donor** | Total unit yang dikirim dari cabang ini |
| **Terima_Dari** | Dari cabang mana stok ini datang 🔵 |
| **Qty_Terima** | Total unit yang berhasil diterima |
| **Sisa_PO_Supplier** | Sisa kebutuhan yang tidak terpenuhi dari pool → harus PO |
| **Skenario** | Kondisi distribusi SKU ini (lihat legenda) |
| **Total Pool** | Total stok yang bisa dibagikan untuk SKU ini |
| **Total Need** | Total kebutuhan semua cabang understock untuk SKU ini |
        """)

    # ── Init Session State ─────────────────────────────────────────────────────
    if "donor_rules" not in st.session_state:
        st.session_state["donor_rules"] = _default_rules()
    if "donor_distance" not in st.session_state:
        st.session_state["donor_distance"] = {k: list(v) for k, v in DEFAULT_DISTANCE_PRIORITY.items()}

    rules    = st.session_state["donor_rules"]
    distance = st.session_state["donor_distance"]

    # ── Pengaturan Donor — di halaman utama ────────────────────────────────────
    st.markdown("---")
    with st.expander("⚙️ Pengaturan Donor Antar Cabang", expanded=False):
        cfg1, cfg2 = st.tabs(["🚫 Aturan Kirim", "📍 Prioritas Jarak"])

        with cfg1:
            st.caption("✅ Centang = **BOLEH** kirim. Kosong = **TIDAK BOLEH**.")
            cols_rule = st.columns(len(ALL_CITIES))
            for i, dcity in enumerate(ALL_CITIES):
                with cols_rule[i]:
                    st.markdown(f"**{dcity}**")
                    for rcity in ALL_CITIES:
                        if dcity == rcity:
                            continue
                        key = f"rule_{dcity}_{rcity}"
                        cur = rules.get(dcity, {}).get(rcity, True)
                        rules[dcity][rcity] = st.checkbox(f"→ {rcity}", value=cur, key=key)
            st.session_state["donor_rules"] = rules

        with cfg2:
            st.caption("Atur urutan prioritas donor per penerima. **Urutan 1 = paling dekat/prioritas utama.**")
            cities_recv = [c for c in ALL_CITIES if c != "SURABAYA"]
            cols_dist = st.columns(len(cities_recv))
            for i, rcity in enumerate(cities_recv):
                with cols_dist[i]:
                    st.markdown(f"**{rcity}**")
                    cur_order = distance.get(rcity, [c for c in ALL_CITIES if c != rcity])
                    others    = [c for c in ALL_CITIES if c != rcity]
                    new_order = []
                    for rank in range(len(others)):
                        remaining = [c for c in others if c not in new_order]
                        if not remaining:
                            break
                        default_choice = cur_order[rank] if rank < len(cur_order) and cur_order[rank] in remaining else remaining[0]
                        sel = st.selectbox(
                            f"Prioritas {rank+1}",
                            options=remaining,
                            index=remaining.index(default_choice),
                            key=f"dist_{rcity}_{rank}",
                        )
                        new_order.append(sel)
                    distance[rcity] = new_order
            st.session_state["donor_distance"] = distance

    rules    = st.session_state["donor_rules"]
    distance = st.session_state["donor_distance"]

    # ── Filter Produk ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Filter Produk")
    c1, c2, c3 = st.columns(3)
    sel_kat   = c1.multiselect("Kategori Barang:", sorted(df["Kategori Barang"].dropna().unique().astype(str)), key="donor_kat")
    sel_brand = c2.multiselect("Brand:",           sorted(df["BRAND Barang"].dropna().unique().astype(str)),    key="donor_brand")
    sel_prod  = c3.multiselect("Nama Produk:",     sorted(df["Nama Barang"].dropna().unique().astype(str)),     key="donor_prod")
    c4, c5    = st.columns(2)
    sel_abc     = c4.multiselect("Kategori ABC:",  sorted(df[KAT_COL].dropna().unique().astype(str)), key="donor_abc")
    only_active = c5.checkbox("Hanya SKU dengan aktivitas donor/terima", value=True, key="donor_active")

    if sel_kat:   df = df[df["Kategori Barang"].astype(str).isin(sel_kat)]
    if sel_brand: df = df[df["BRAND Barang"].astype(str).isin(sel_brand)]
    if sel_prod:  df = df[df["Nama Barang"].astype(str).isin(sel_prod)]
    if sel_abc:   df = df[df[KAT_COL].isin(sel_abc)]

    # ── Hitung ────────────────────────────────────────────────────────────────
    st.markdown("---")
    with st.spinner("⏳ Menghitung distribusi donor..."):
        donor_df = _run_donor_calc(df, rules, distance)

    if donor_df.empty:
        st.info("Tidak ada data untuk diproses.")
        st.stop()

    if only_active:
        active_skus = donor_df[(donor_df["Donor_Ke"] != "-") | (donor_df["Terima_Dari"] != "-")]["No. Barang"].unique()
        ddisp = donor_df[donor_df["No. Barang"].isin(active_skus)].copy()
    else:
        ddisp = donor_df.copy()

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.subheader("📊 Ringkasan Hasil")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("SKU ada donor",         donor_df[donor_df["Donor_Ke"] != "-"]["No. Barang"].nunique())
    m2.metric("SKU menerima transfer", donor_df[donor_df["Terima_Dari"] != "-"]["No. Barang"].nunique())
    m3.metric("Total unit didonorkan", f"{int(donor_df['Qty_Donor'].sum()):,}")
    m4.metric("Total unit diterima",   f"{int(donor_df['Qty_Terima'].sum()):,}")
    m5.metric("Sisa → butuh PO",       f"{int(donor_df['Sisa_PO_Supplier'].sum()):,}")

    # ── Skenario counts ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Distribusi Skenario per SKU")
    sku_ske = donor_df.drop_duplicates("No. Barang")["Skenario"].value_counts()
    skcols  = st.columns(max(len(sku_ske), 1))
    for i, (lbl, cnt) in enumerate(sku_ske.items()):
        bg = _SKENARIO_COLOR.get(lbl, "#eee")
        skcols[i].markdown(
            f"<div style='background:{bg};padding:10px;border-radius:8px;text-align:center'>"
            f"<b style='font-size:0.8em'>{lbl}</b><br>"
            f"<span style='font-size:1.6em;font-weight:bold'>{cnt}</span> SKU</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Detail per Cabang",
        "📦 Rekap per SKU",
        "📊 Tabel Gabungan",
        "🔀 Matriks Transfer",
    ])

    DISP_COLS = [
        "No. Barang", "Kategori Barang", "BRAND Barang", "Nama Barang",
        "City", "Kategori ABC",
        "Stock Cabang", "Min Stock", "Max Stock",
        "Status Stock", "% Stock", "Add Stock", "Qty_Bisa_Donor",
        "Donor_Ke", "Qty_Donor",
        "Terima_Dari", "Qty_Terima",
        "Sisa_PO_Supplier", "Skenario",
    ]
    ecols = [c for c in DISP_COLS if c in ddisp.columns]
    FMT   = {"% Stock": "{:.1f}", "Stock Cabang": "{:.0f}", "Min Stock": "{:.0f}",
              "Max Stock": "{:.0f}", "Add Stock": "{:.0f}", "Qty_Bisa_Donor": "{:.0f}",
              "Qty_Donor": "{:.0f}", "Qty_Terima": "{:.0f}", "Sisa_PO_Supplier": "{:.0f}"}

    # Tab 1
    with tab1:
        st.subheader("Detail per Cabang")
        st.caption("🟢 Kolom **Donor_Ke** = cabang ini mengirim  |  🔵 Kolom **Terima_Dari** = cabang ini menerima")
        for city in sorted(ddisp["City"].unique()):
            cdf   = ddisp[ddisp["City"] == city][ecols].copy()
            n_act = ((cdf["Donor_Ke"] != "-") | (cdf["Terima_Dari"] != "-")).sum()
            with st.expander(f"📍 {city}  —  {n_act} SKU aktif dari {len(cdf)}", expanded=(n_act > 0)):
                styled = cdf.style.format(FMT, na_rep="-")
                for col in cdf.columns:
                    if   col == "Status Stock":   styled = styled.map(highlight_status_stock,    subset=[col])
                    elif col == "Kategori ABC":   styled = styled.map(highlight_kategori_abc_log, subset=[col])
                    elif col == "Skenario":       styled = styled.map(_hl_ske,   subset=[col])
                    elif col == "Donor_Ke":       styled = styled.map(_hl_donor, subset=[col])
                    elif col == "Terima_Dari":    styled = styled.map(_hl_recv,  subset=[col])
                st.dataframe(styled, use_container_width=True)

    # Tab 2
    with tab2:
        st.subheader("Rekap per SKU")
        st.caption("1 baris = 1 SKU. Lihat siapa yang kirim, siapa yang terima, dan berapa sisa yang harus PO.")
        KEYS = ["No. Barang", "Kategori Barang", "BRAND Barang", "Nama Barang"]
        rows2 = []
        for sku, grp in ddisp.groupby("No. Barang"):
            f    = grp.iloc[0]
            meta = {k: f.get(k, "") for k in KEYS}
            da   = grp[(grp["Donor_Ke"] != "-") & (grp["Qty_Donor"] > 0)]
            ra   = grp[(grp["Terima_Dari"] != "-") & (grp["Qty_Terima"] > 0)]
            meta.update({
                "Skenario":          grp["Skenario"].iloc[0],
                "Total Need":        int(grp["Total_Need"].iloc[0]),
                "Total Pool":        int(grp["Total_Pool"].iloc[0]),
                "Terpenuhi":         int(grp["Qty_Terima"].sum()),
                "Sisa PO Supplier":  int(grp["Sisa_PO_Supplier"].sum()),
                "Donor (kirim ke)":  " | ".join(f"{r['City']}→{r['Donor_Ke']} ({int(r['Qty_Donor'])} unit)" for _, r in da.iterrows()) or "-",
                "Penerima (dari)":   " | ".join(f"{r['City']}←{r['Terima_Dari']} ({int(r['Qty_Terima'])} unit)" for _, r in ra.iterrows()) or "-",
            })
            rows2.append(meta)
        df2 = pd.DataFrame(rows2)
        if not df2.empty:
            st.dataframe(df2.style.map(_hl_ske, subset=["Skenario"]), use_container_width=True, height=500)
        else:
            st.info("Tidak ada SKU aktif.")

    # Tab 3: Tabel Gabungan Pivot
    with tab3:
        st.subheader("Tabel Gabungan — Semua Cabang dalam Satu Baris per SKU")
        st.caption("Format mirip Tabel Gabungan di V2. Setiap baris = 1 SKU, kolom = info tiap cabang.")
        KEYS = ["No. Barang", "Kategori Barang", "BRAND Barang", "Nama Barang"]
        CITY_SHORT = {"BALI":"BALI","JAKARTA":"JKT","JOGJA":"JOG","MALANG":"MLG","SEMARANG":"SMG","SURABAYA":"SBY"}
        M_COLS = ["Stock Cabang","Min Stock","Max Stock","Add Stock","Status Stock",
                  "% Stock","Qty_Bisa_Donor","Donor_Ke","Qty_Donor","Terima_Dari","Qty_Terima","Sisa_PO_Supplier"]
        mex = [m for m in M_COLS if m in ddisp.columns]

        try:
            piv = ddisp.pivot_table(index=KEYS, columns="City", values=mex, aggfunc="first")
            piv.columns = [f"{CITY_SHORT.get(city,city)}_{metric}" for metric, city in piv.columns]
            piv = piv.reset_index()

            sku_agg = ddisp.groupby("No. Barang").agg(
                All_Total_Need    = ("Total_Need",       "first"),
                All_Total_Pool    = ("Total_Pool",       "first"),
                All_Qty_Terima    = ("Qty_Terima",       "sum"),
                All_Sisa_PO       = ("Sisa_PO_Supplier", "sum"),
                All_Skenario      = ("Skenario",         "first"),
            ).reset_index()
            piv = piv.merge(sku_agg, on="No. Barang", how="left")

            non_sby = [c for c in piv.columns if c not in KEYS and not c.startswith("SBY_") and not c.startswith("All_")]
            sby_c   = [c for c in piv.columns if c.startswith("SBY_")]
            all_c   = [c for c in piv.columns if c.startswith("All_")]
            piv     = piv[KEYS + non_sby + sby_c + all_c]

            for col in piv.columns:
                if col in KEYS: continue
                if pd.api.types.is_numeric_dtype(piv[col]):
                    piv[col] = piv[col].fillna(0).astype(int)
                else:
                    piv[col] = piv[col].fillna("-")

            ccfg = {c: st.column_config.NumberColumn(format="%.0f")
                    for c in piv.columns if c not in KEYS and pd.api.types.is_numeric_dtype(piv[c])}

            style_piv = piv.style
            if "All_Skenario" in piv.columns:
                style_piv = style_piv.map(_hl_ske, subset=["All_Skenario"])

            st.dataframe(style_piv, column_config=ccfg, use_container_width=True, height=500)

            with st.expander("ℹ️ Keterangan singkatan kolom"):
                st.markdown("""
| Prefix | Cabang |
|---|---|
| `JKT_` | Jakarta | `SMG_` | Semarang |
| `JOG_` | Jogja | `MLG_` | Malang |
| `BALI_` | Bali | `SBY_` | Surabaya |

| Kolom All_ | Arti |
|---|---|
| `All_Total_Need` | Total kebutuhan semua cabang understock (SKU ini) |
| `All_Total_Pool` | Total stok tersedia untuk dibagi |
| `All_Qty_Terima` | Total yang berhasil didistribusikan |
| `All_Sisa_PO` | Total yang masih harus PO ke supplier |
| `All_Skenario` | Skenario distribusi SKU ini |
                """)
        except Exception as e:
            st.warning(f"Gagal membuat tabel gabungan: {e}")

    # Tab 4: Matriks
    with tab4:
        st.subheader("🔀 Matriks Transfer Antar Cabang")
        st.caption("Baris = pengirim | Kolom = penerima | Angka = total unit yang ditransfer")

        cits   = sorted(ddisp["City"].unique())
        matrix = pd.DataFrame(0, index=cits, columns=cits, dtype=int)
        for _, row in ddisp.iterrows():
            if row["Donor_Ke"] == "-" or row["Qty_Donor"] == 0:
                continue
            dests = [d.strip() for d in str(row["Donor_Ke"]).split(",") if d.strip()]
            each  = int(row["Qty_Donor"]) // max(len(dests), 1)
            for dest in dests:
                if dest in matrix.columns:
                    matrix.loc[row["City"], dest] += each

        matrix["► TOTAL KIRIM"]      = matrix.sum(axis=1)
        matrix.loc["▼ TOTAL TERIMA"] = matrix.sum(axis=0)

        def _cm(v):
            return "background-color: #cce5ff; font-weight: bold" if isinstance(v, (int, float, np.integer)) and v > 0 else ""
        st.dataframe(matrix.style.map(_cm), use_container_width=True)

        st.markdown("---")
        st.subheader("🚫 Aturan Aktif Saat Ini")
        rule_rows = [
            {"Pengirim": d, "Penerima": r, "Status": "✅ Boleh" if rules.get(d, {}).get(r, True) else "🚫 Tidak Boleh"}
            for d in ALL_CITIES for r in ALL_CITIES if d != r
        ]
        rdf = pd.DataFrame(rule_rows)
        tb  = rdf[rdf["Status"] == "🚫 Tidak Boleh"]
        if tb.empty:
            st.success("Semua rute saat ini diizinkan.")
        else:
            st.dataframe(tb, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📍 Prioritas Jarak Aktif Saat Ini")
        dist_rows = []
        for rcity in ALL_CITIES:
            if rcity == "SURABAYA": continue
            order = distance.get(rcity, [])
            for rank, dcity in enumerate(order):
                dist_rows.append({"Penerima": rcity, "Prioritas": rank+1, "Donor": dcity})
        st.dataframe(pd.DataFrame(dist_rows), use_container_width=True, hide_index=True)

    # ── Download ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Unduh Hasil Analisis Donor")

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        ddisp[ecols].to_excel(writer, sheet_name="Detail per Cabang", index=False)
        if not df2.empty:
            df2.to_excel(writer, sheet_name="Rekap per SKU", index=False)
        try:
            piv.to_excel(writer, sheet_name="Tabel Gabungan", index=False)
        except Exception:
            pass
        matrix.to_excel(writer, sheet_name="Matriks Transfer")
        for city in sorted(ddisp["City"].unique()):
            cdf = ddisp[ddisp["City"] == city][ecols]
            if not cdf.empty:
                cdf.to_excel(writer, sheet_name=city[:31], index=False)

    st.download_button(
        "📥 Unduh Excel — Analisis Donor",
        data=output.getvalue(),
        file_name="Analisis_Donor_Stock.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ── Legenda ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📖 Legenda Skenario")
    lg1, lg2, lg3 = st.columns(3)
    lg1.success("**0 - TIDAK ADA KEBUTUHAN**\nSemua cabang sudah aman.")
    lg2.info("**2 - SBY CUKUP**\nSisa SBY cukup untuk semua.")
    lg3.warning("**3 - SBY TERBATAS + ADA DONOR**\nSBY kurang, dibantu donor cabang.")
    lg4, lg5, lg6 = st.columns(3)
    lg4.warning("**4 - HANYA DONOR CABANG**\nSBY tidak bisa kirim, donor cabang aktif.")
    lg5.error("**5 - POOL TIDAK CUKUP**\nPool ada tapi tidak cukup → sebagian PO.")
    lg6.error("**1 - KURANG**\nTidak ada pool sama sekali → semua PO.")
