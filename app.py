import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FOTO_B64 = "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"
WAJIB = {"Data_Raw": ["Transaction_ID","Transaction_Date","Customer_ID","Product_ID","Product_Name","Quantity","Unit_Price","Discount_Rate","Line_Total","Region","Customer_Type","Payment_Method"],
         "Data_Produk": ["Product_ID","Product_Name","Category","Reference_Unit_Price_Rp"],
         "Data_Pelanggan": ["Customer_ID","Region","Customer_Type"]}
NAMA, NPM = "Allesandro Marhaen Saunoah", "51230043"
MERAH = "#C62E1A"

st.set_page_config(page_title="BI Toko | " + NAMA, page_icon="📊", layout="wide")
st.markdown("""<style>
[data-testid="stMetricValue"] {font-size:1.7rem}
[data-testid="stMetricValue"] > div {overflow:visible !important; text-overflow:clip !important;
                                     white-space:normal !important; word-break:break-word}
</style>""", unsafe_allow_html=True)


# ───────────────────────── DATA ─────────────────────────
@st.cache_data(show_spinner="Memuat & membersihkan data...")
def load_all(data: bytes):
    xl = pd.ExcelFile(io.BytesIO(data))
    raw = xl.parse("Data_Raw")
    prod = xl.parse("Data_Produk")
    cust = xl.parse("Data_Pelanggan")
    raw["Transaction_Date"] = pd.to_datetime(raw["Transaction_Date"], errors="coerce")
    log = [("Data mentah (baris)", len(raw), "-")]

    d = raw.copy()  # Data_Raw tidak ditimpa
    n = len(d)
    d = d.drop_duplicates()
    log.append(("Duplikasi baris", n - len(d), "Dihapus"))

    # teks tidak konsisten → kembalikan dari katalog produk
    pmap = prod.set_index("Product_Name")
    d["Product_ID"] = d["Product_ID"].fillna(d["Product_Name"].map(pmap["Product_ID"]))
    d["Category"] = d["Product_ID"].map(prod.set_index("Product_ID")["Category"])
    d["Payment_Method"] = (d["Payment_Method"].str.strip().str.title()
                           .replace({"Qris": "QRIS"}).fillna("Tidak Diketahui"))

    # Customer_ID kosong: isi dari baris lain pada transaksi yang sama
    miss = d["Customer_ID"].isna().sum()
    d["Customer_ID"] = d.groupby("Transaction_ID")["Customer_ID"].transform(lambda s: s.ffill().bfill())
    log.append(("Customer_ID kosong", miss, f"Diisi dari transaksi sama; sisa {d['Customer_ID'].isna().sum()} dihapus"))
    d = d.dropna(subset=["Customer_ID"])

    # Region & Customer_Type dari master pelanggan
    c = cust.set_index("Customer_ID")
    log.append(("Region/Customer_Type kosong", int(d["Region"].isna().sum() + d["Customer_Type"].isna().sum()),
                "Diimputasi dari Data_Pelanggan"))
    d["Region"] = d["Region"].fillna(d["Customer_ID"].map(c["Region"]))
    d["Customer_Type"] = d["Customer_Type"].fillna(d["Customer_ID"].map(c["Customer_Type"]))

    # harga tidak valid → harga referensi katalog
    ref = d["Product_ID"].map(prod.set_index("Product_ID")["Reference_Unit_Price_Rp"])
    bad_p = d["Unit_Price"].isna() | (d["Unit_Price"] <= 0) | (d["Unit_Price"] != ref)
    log.append(("Unit_Price kosong/tidak wajar", int(bad_p.sum()), "Diganti harga referensi"))
    d["Unit_Price"] = ref

    # quantity tidak valid → turunkan dari Line_Total, jika tidak mungkin dihapus
    est = (d["Line_Total"] / (d["Unit_Price"] * (1 - d["Discount_Rate"]))).round()
    bad_q = d["Quantity"].isna() | (d["Quantity"] <= 0) | (d["Quantity"] > 20)
    fix = bad_q & est.between(1, 20)
    d.loc[fix, "Quantity"] = est[fix]
    log.append(("Quantity kosong/tidak valid", int(bad_q.sum()),
                f"{int(fix.sum())} diperbaiki dari Line_Total, {int((bad_q & ~fix).sum())} dihapus"))
    d = d[~(bad_q & ~fix)]

    d["Line_Total"] = d["Quantity"] * d["Unit_Price"] * (1 - d["Discount_Rate"])
    d["Bulan"] = d["Transaction_Date"].dt.to_period("M").dt.to_timestamp()
    d = d.reset_index(drop=True)
    log.append(("Data bersih (baris)", len(d), "-"))
    return raw, d, prod, cust, pd.DataFrame(log, columns=["Tahap", "Jumlah", "Keputusan"])


@st.cache_data
def make_rfm(d):
    ref = d["Transaction_Date"].max().normalize() + pd.Timedelta(days=1)
    r = d.groupby("Customer_ID").agg(
        Recency=("Transaction_Date", lambda s: (ref - s.max()).days),
        Frequency=("Transaction_ID", "nunique"), Monetary=("Line_Total", "sum"))
    return r, ref


@st.cache_data
def kmeans_eval(rfm, ks=tuple(range(2, 9))):
    X = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    out = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=42).fit(X)
        out.append((k, km.inertia_, silhouette_score(X, km.labels_)))
    return pd.DataFrame(out, columns=["k", "Inertia", "Silhouette"])


def segment(rfm, k):
    X = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    r = rfm.copy()
    r["Cluster"] = KMeans(k, n_init=10, random_state=42).fit_predict(X)
    prof = r.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].median()
    names = {}
    for cl, row in prof.iterrows():
        if row.Frequency >= prof.Frequency.median() and row.Monetary >= prof.Monetary.median():
            names[cl] = "Loyal / Bernilai Tinggi" if row.Recency <= prof.Recency.median() else "Bernilai Tinggi - Berisiko"
        elif row.Recency > prof.Recency.median():
            names[cl] = "Berisiko Tidak Kembali"
        else:
            names[cl] = "Potensial / Baru"
    # nama harus unik
    seen = {}
    for cl in names:
        seen[names[cl]] = seen.get(names[cl], 0) + 1
        if seen[names[cl]] > 1:
            names[cl] += f" ({seen[names[cl]]})"
    r["Segmen"] = r["Cluster"].map(names)
    return r


@st.cache_data
def run_apriori(d, sup, conf):
    bsk = pd.crosstab(d["Transaction_ID"], d["Product_Name"]).gt(0)
    fi = apriori(bsk, min_support=sup, use_colnames=True)
    if fi.empty:
        return bsk, fi, pd.DataFrame()
    ru = association_rules(fi, metric="confidence", min_threshold=conf)
    if not ru.empty:
        ru = ru[ru["lift"] > 1].copy()
        ru["Aturan"] = ru["antecedents"].map(lambda s: ", ".join(s)) + " → " + ru["consequents"].map(lambda s: ", ".join(s))
        ru = ru.sort_values(["lift", "confidence"], ascending=False)
    return bsk, fi, ru


rupiah = lambda x: "Rp {:,.0f}".format(x).replace(",", ".")

# ───────────────────────── HALAMAN DEPAN ─────────────────────────
if "masuk" not in st.session_state:
    st.session_state.masuk = False

if not st.session_state.masuk:
    foto = FOTO_B64
    st.markdown(f"""
    <style>
    [data-testid="stHeader"], [data-testid="stSidebar"] {{display:none}}
    .stApp {{background:linear-gradient(160deg,{MERAH} 0%,#7d1509 100%)}}
    .kartu {{max-width:440px;margin:4vh auto 1rem;background:#fff;border-radius:24px;padding:36px 28px;
             text-align:center;box-shadow:0 20px 60px rgba(0,0,0,.35)}}
    .kartu img {{width:190px;height:190px;border-radius:50%;object-fit:cover;object-position:top;
                 border:6px solid {MERAH}}}
    .kartu h1 {{font-size:1.5rem;margin:18px 0 4px;color:#222}}
    .kartu .npm {{display:inline-block;background:{MERAH};color:#fff;border-radius:999px;padding:4px 18px;
                  font-weight:700;letter-spacing:2px}}
    .kartu small {{display:block;margin-top:18px;color:#666;line-height:1.5}}
    </style>
    <div class="kartu">
      <img src="data:image/jpeg;base64,{foto}">
      <h1>{NAMA}</h1><span class="npm">NPM {NPM}</span>
      <small><b>Sistem Business Intelligence Toko</b><br>Data Mining &amp; BI · Analisis Transaksi 2025<br>
      Universitas Timor</small>
    </div>""", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1, 1])
    if mid.button("Masuk ke Sistem →", type="primary", width="stretch"):
        st.session_state.masuk = True
        st.rerun()
    st.stop()

# ───────────────────────── UPLOAD DATASET ─────────────────────────
if st.session_state.get("data_bytes") is None:
    st.title("📁 Upload Dataset")
    st.write("Unggah file Excel (.xlsx) yang berisi sheet **Data_Raw**, **Data_Produk**, dan **Data_Pelanggan**.")
    up = st.file_uploader("Pilih file Excel", type=["xlsx"])
    if up is not None:
        try:
            xl = pd.ExcelFile(io.BytesIO(up.getvalue()))
            salah = []
            for sh, cols in WAJIB.items():
                if sh not in xl.sheet_names:
                    salah.append(f"Sheet **{sh}** tidak ditemukan.")
                else:
                    kurang = set(cols) - set(xl.parse(sh, nrows=1).columns)
                    if kurang:
                        salah.append(f"Sheet **{sh}** kekurangan kolom: {', '.join(sorted(kurang))}")
            if salah:
                st.error("File tidak sesuai format:\n\n" + "\n\n".join("- " + x for x in salah))
            else:
                st.session_state.data_bytes = up.getvalue()
                st.session_state.nama_file = up.name
                st.rerun()
        except Exception as e:
            st.error(f"File tidak dapat dibaca: {e}")
    if st.button("⬅ Kembali ke halaman depan"):
        st.session_state.masuk = False
        st.rerun()
    st.stop()

# ───────────────────────── SISTEM ─────────────────────────
raw, clean, prod, cust, log = load_all(st.session_state.data_bytes)
rfm, ref_date = make_rfm(clean)

with st.sidebar:
    st.markdown(f"**{NAMA}**  \nNPM {NPM}")
    halaman = st.radio("Menu", ["📊 Dashboard BI", "🔍 Business Understanding & EDA", "🧹 Preprocessing",
                                "🛒 Apriori (Market Basket)", "👥 K-Means (Segmentasi)", "✅ Rekomendasi Keputusan"])
    st.caption(f"📄 {st.session_state.get('nama_file', 'dataset')}")
    if st.button("🔄 Ganti dataset"):
        st.session_state.data_bytes = None
        st.cache_data.clear()
        st.rerun()
    if st.button("⬅ Keluar ke halaman depan"):
        st.session_state.masuk = False
        st.rerun()

seg_k = st.session_state.get("k", None)
ev = kmeans_eval(rfm)
if seg_k is None:
    seg_k = int(ev.loc[ev["Silhouette"].idxmax(), "k"])
seg = segment(rfm, seg_k)

# ── Dashboard
if halaman.startswith("📊"):
    st.title("📊 Dashboard Business Intelligence")
    with st.sidebar:
        st.subheader("Filter")
        tmin, tmax = clean["Transaction_Date"].min().date(), clean["Transaction_Date"].max().date()
        tgl = st.date_input("Rentang tanggal", (tmin, tmax), min_value=tmin, max_value=tmax)
        kat = st.multiselect("Kategori", sorted(clean["Category"].unique()))
        sg = st.multiselect("Segmen pelanggan", sorted(seg["Segmen"].unique()))
    f = clean.copy()
    if len(tgl) == 2:
        f = f[f["Transaction_Date"].between(pd.Timestamp(tgl[0]), pd.Timestamp(tgl[1]) + pd.Timedelta(days=1))]
    if kat:
        f = f[f["Category"].isin(kat)]
    if sg:
        f = f[f["Customer_ID"].isin(seg.index[seg["Segmen"].isin(sg)])]
    if f.empty:
        st.warning("Tidak ada data untuk filter ini.")
        st.stop()

    bs = f.groupby("Transaction_ID")["Product_ID"].nunique()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Penjualan", rupiah(f["Line_Total"].sum()))
    k2.metric("Jumlah Transaksi", f"{f['Transaction_ID'].nunique():,}")
    k3.metric("Pelanggan Unik", f"{f['Customer_ID'].nunique():,}")
    k4.metric("Rata-rata Basket Size", f"{bs.mean():.2f}")

    a, b = st.columns(2)
    m = f.groupby("Bulan")["Line_Total"].sum().reset_index()
    a.plotly_chart(px.line(m, x="Bulan", y="Line_Total", markers=True, title="Tren Penjualan Bulanan",
                           color_discrete_sequence=[MERAH]), width="stretch")
    top = f.groupby("Product_Name")["Line_Total"].sum().nlargest(10).reset_index()
    b.plotly_chart(px.bar(top, x="Line_Total", y="Product_Name", orientation="h", title="Top 10 Produk (omzet)",
                          color_discrete_sequence=[MERAH]).update_yaxes(autorange="reversed"), width="stretch")
    c, e = st.columns(2)
    kt = f.groupby("Category")["Line_Total"].sum().reset_index()
    c.plotly_chart(px.pie(kt, names="Category", values="Line_Total", hole=.45, title="Kontribusi Kategori"),
                   width="stretch")
    sp = seg.groupby("Segmen").agg(Pelanggan=("Monetary", "size"), Monetary=("Monetary", "median")).reset_index()
    e.plotly_chart(px.bar(sp, x="Segmen", y="Pelanggan", title="Profil Segmen K-Means", color="Segmen"),
                   width="stretch")
    _, _, ru = run_apriori(clean, 0.01, 0.1)
    st.subheader("Hasil Apriori (Top 5 berdasarkan lift)")
    if ru.empty:
        st.info("Tidak ada aturan pada parameter default.")
    else:
        st.dataframe(ru.head(5)[["Aturan", "support", "confidence", "lift"]].round(3), hide_index=True,
                     width="stretch")

# ── EDA
elif halaman.startswith("🔍"):
    st.title("🔍 Business Understanding & EDA")
    st.markdown("**Pertanyaan bisnis:**\n1. Produk apa yang sering dibeli bersama?\n"
                "2. Segmen pelanggan mana yang paling bernilai?\n3. Kapan penjualan meningkat/menurun?")
    st.subheader("Ringkasan data mentah")
    c = st.columns(4)
    c[0].metric("Baris × Kolom", f"{raw.shape[0]:,} × {raw.shape[1]}")
    c[1].metric("Transaksi unik", f"{raw['Transaction_ID'].nunique():,}")
    c[2].metric("Pelanggan / Produk unik", f"{raw['Customer_ID'].nunique():,} / {raw['Product_ID'].nunique()}")
    c[3].metric("Rentang tanggal", f"{raw['Transaction_Date'].min():%d %b} – {raw['Transaction_Date'].max():%d %b %Y}")
    st.write("Total penjualan sementara (mentah):", rupiah(raw["Line_Total"].sum()))
    q = pd.DataFrame({"Missing": raw.isna().sum(), "Tipe data": raw.dtypes.astype(str)})
    st.dataframe(q, width="stretch")
    st.write("Duplikasi baris:", int(raw.duplicated().sum()), "| Quantity ≤0 atau >20:",
             int(((raw.Quantity <= 0) | (raw.Quantity > 20)).sum()), "| Unit_Price ≤0 atau >200.000:",
             int(((raw.Unit_Price <= 0) | (raw.Unit_Price > 200000)).sum()))
    a, b = st.columns(2)
    a.plotly_chart(px.box(raw[raw.Quantity.between(-5, 1000)], y="Quantity", log_y=False,
                          title="Outlier Quantity (mentah)"), width="stretch")
    b.plotly_chart(px.histogram(clean.groupby("Transaction_ID")["Product_ID"].nunique(), title="Distribusi Basket Size",
                                labels={"value": "Produk unik/transaksi"}, color_discrete_sequence=[MERAH]),
                   width="stretch")
    st.subheader("Insight awal")
    kt = clean.groupby("Category")["Line_Total"].sum().sort_values(ascending=False)
    st.markdown(f"- Kategori omzet terbesar: **{kt.index[0]}** ({kt.iloc[0] / kt.sum():.0%} dari total).\n"
                f"- Rata-rata basket size **{clean.groupby('Transaction_ID')['Product_ID'].nunique().mean():.2f}** produk/transaksi.\n"
                f"- Wilayah dengan omzet tertinggi: **{clean.groupby('Region')['Line_Total'].sum().idxmax()}**.")

# ── Preprocessing
elif halaman.startswith("🧹"):
    st.title("🧹 Preprocessing Data")
    st.caption("Data_Raw tidak ditimpa; semua perbaikan dilakukan pada salinan kerja.")
    st.dataframe(log, hide_index=True, width="stretch")
    bsz = clean.groupby("Transaction_ID")["Product_ID"].nunique().rename("basket_size")
    st.write(f"**basket_size** = produk unik per Transaction_ID (rata-rata {bsz.mean():.2f}).")
    st.write(f"**RFM**: tanggal acuan = {ref_date:%d %B %Y} (1 hari setelah transaksi terakhir). "
             "Recency = hari sejak transaksi terakhir, Frequency = transaksi unik, Monetary = total penjualan bersih.")
    st.dataframe(rfm.describe().round(1), width="stretch")
    st.info("Standardisasi (z-score) dipakai karena K-Means berbasis jarak; Monetary (jutaan) akan mendominasi "
            "Recency/Frequency tanpa scaling. Data ±1.500 pelanggan sudah ringan; untuk data besar gunakan agregasi "
            "per pelanggan (seperti RFM ini), sampling, atau MiniBatchKMeans.")
    st.download_button("⬇ Unduh Data_Clean (CSV)", clean.to_csv(index=False).encode(), "data_clean.csv")

# ── Apriori
elif halaman.startswith("🛒"):
    st.title("🛒 Apriori / Market Basket Analysis")
    c1, c2 = st.columns(2)
    sup = c1.slider("Minimum support", 0.005, 0.10, 0.01, 0.005)
    conf = c2.slider("Minimum confidence", 0.05, 0.9, 0.10, 0.05)
    bsk, fi, ru = run_apriori(clean, sup, conf)
    st.write(f"Matriks transaksi × produk: **{bsk.shape[0]:,} × {bsk.shape[1]}** | Frequent itemsets: **{len(fi)}**")
    st.caption("Support rendah dipilih karena 30 produk tersebar merata (tiap pasangan jarang muncul). "
               "Aturan diurutkan berdasarkan lift, lalu confidence; hanya lift > 1 yang ditampilkan.")
    if ru.empty:
        st.warning("Tidak ada aturan. Turunkan support/confidence.")
    else:
        t = ru.head(10)[["Aturan", "support", "confidence", "lift"]].round(3)
        st.dataframe(t, hide_index=True, width="stretch")
        st.plotly_chart(px.scatter(ru.head(30), x="support", y="confidence", size="lift", color="lift",
                                   hover_name="Aturan", title="Aturan Asosiasi (ukuran = lift)"),
                        width="stretch")
        r0 = ru.iloc[0]
        st.markdown(f"**Rekomendasi cross-selling/bundling:** paket bundling *{r0['Aturan']}* "
                    f"(lift {r0['lift']:.2f}); letakkan produk-produk pada aturan teratas berdekatan di rak. "
                    "Ingat: aturan asosiasi bukan bukti sebab-akibat.")

# ── K-Means
elif halaman.startswith("👥"):
    st.title("👥 K-Means — Segmentasi Pelanggan")
    st.write(f"Satu baris = satu pelanggan ({len(rfm):,} pelanggan), fitur RFM terstandardisasi.")
    a, b = st.columns(2)
    a.plotly_chart(px.line(ev, x="k", y="Inertia", markers=True, title="Elbow Method"), width="stretch")
    b.plotly_chart(px.line(ev, x="k", y="Silhouette", markers=True, title="Silhouette Score"), width="stretch")
    best = int(ev.loc[ev["Silhouette"].idxmax(), "k"])
    k = st.slider("Pilih k", 2, 8, seg_k)
    st.session_state["k"] = k
    seg = segment(rfm, k)
    st.caption(f"k dengan silhouette tertinggi: {best}. Pilih k dengan mempertimbangkan metrik dan keterbacaan bisnis.")
    prof = seg.groupby("Segmen").agg(Pelanggan=("Monetary", "size"), Recency_median=("Recency", "median"),
                                     Frequency_median=("Frequency", "median"),
                                     Monetary_median=("Monetary", "median"), Monetary_rata2=("Monetary", "mean")).round(0)
    st.dataframe(prof, width="stretch")
    st.plotly_chart(px.scatter(seg.reset_index(), x="Frequency", y="Monetary", color="Segmen", size_max=8,
                               hover_data=["Customer_ID", "Recency"], title="Scatter Segmen (Frequency vs Monetary)"),
                    width="stretch")
    st.subheader("Strategi pemasaran")
    st.markdown("- **Loyal / Bernilai Tinggi:** program member, reward poin, akses awal produk baru.\n"
                "- **Potensial / Baru:** voucher transaksi kedua, bundling produk kebutuhan harian.\n"
                "- **Berisiko Tidak Kembali / Bernilai Tinggi - Berisiko:** kampanye win-back (diskon terbatas, pengingat WhatsApp).")

# ── Rekomendasi
else:
    st.title("✅ Rekomendasi Keputusan")
    _, _, ru = run_apriori(clean, 0.01, 0.1)
    risk = seg[seg["Segmen"].str.contains("Berisiko")]
    tkat = clean.groupby("Category")["Line_Total"].sum().idxmax()
    r0 = ru.iloc[0]["Aturan"] if not ru.empty else "(belum ada aturan)"
    recs = pd.DataFrame([
        ["1. Bundling & penempatan rak", f"Aturan teratas: {r0}", "Paket bundling + rak berdekatan",
         "Basket size, penjualan paket", "Estimasi: kenaikan 3–5% omzet produk terkait (ASUMSI, belum diuji)"],
        ["2. Retensi pelanggan berisiko", f"{len(risk):,} pelanggan segmen berisiko ({len(risk) / len(seg):.0%})",
         "Kampanye win-back diskon terbatas", "Repeat rate 60 hari, redemption voucher",
         "Estimasi: 10% kembali bertransaksi (ASUMSI)"],
        ["3. Fokus pemasaran kategori utama", f"Kategori omzet terbesar: {tkat}", "Promo terjadwal pada bulan puncak",
         "Omzet bulanan kategori", "Dampak aktual diukur via A/B / periode pembanding"]],
        columns=["Rekomendasi", "Bukti data (FAKTA)", "Tindakan", "KPI keberhasilan", "Estimasi manfaat & asumsi"])
    st.dataframe(recs, hide_index=True, width="stretch")
    st.warning("Pisahkan **fakta** (kolom bukti data), **estimasi berbasis asumsi** (kolom estimasi), dan **dampak aktual** "
               "(baru diketahui setelah implementasi + uji, mis. membandingkan kelompok kontrol).")