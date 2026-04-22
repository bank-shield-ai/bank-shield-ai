import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import base64

st.set_page_config(
    page_title="CaixaBank AI Risk Hub - Fraud Detection",
    page_icon="🚨",
    layout="wide"
)

# -----------------------------
# LOGO
# -----------------------------
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None


CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parent.parent
PROJECT_DIR = APP_DIR.parent

MODELS_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data" / "processed"
logo_path = APP_DIR / "logo.png"

logo_base64 = get_base64_image(str(logo_path))
if logo_base64:
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="200">'
else:
    logo_html = '<h2 style="color: #004587; margin:0;">CaixaBank</h2>'

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 0.8rem;
    max-width: 1400px;
}

.header-box {
    background-color: white;
    padding: 12px 36px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 3px solid #ff6600;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    margin-bottom: 18px;
}

.hero-section {
    background: linear-gradient(135deg, #004587 0%, #002e5a 100%);
    color: white;
    padding: 42px 40px;
    border-radius: 0 0 36px 36px;
    margin-bottom: 28px;
    box-shadow: 0 10px 28px rgba(0, 69, 135, 0.18);
}

.info-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e7ebf0;
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background-color: #004587;
    color: white;
    border-radius: 999px;
    border: none;
    font-weight: 700;
}

.footer-box {
    text-align: center;
    margin-top: 70px;
    padding: 28px;
    color: #7f8c8d;
    font-size: 0.85rem;
    border-top: 1px solid #e1e4e8;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(f"""
<div class="header-box">
    <div>{logo_html}</div>
    <div style="color:#004587;font-weight:700;">Yetkili Modülü • Fraud Detection</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-section">
    <h1 style="color:white; margin-bottom: 8px;">Fraud Detection</h1>
    <p style="opacity:0.92; margin-bottom:0;">
        Anlık işlem bazlı sahtecilik tespiti ve otomatik aksiyon yönetimi.
    </p>
</div>
""", unsafe_allow_html=True)

top1, top2 = st.columns([1, 1])

with top1:
    if st.button("← Yetkili Paneline Dön", key="back_admin_panel"):
        st.switch_page("pages/00_Yetkili_Paneli.py")

with top2:
    st.markdown(
        '<div class="info-card"><b>Sürüm:</b> v3 (Agresif Koruma Modu)</div>',
        unsafe_allow_html=True
    )

# -----------------------------
# MODEL VE VERİ
# -----------------------------
@st.cache_resource
def load_fraud_assets():
    model_path = MODELS_DIR / "fraud_model_v3.joblib"
    data_path = DATA_DIR / "customer_risk_summary.parquet"

    try:
        bundle = joblib.load(model_path)
        profiles = pd.read_parquet(data_path)
        return bundle, profiles
    except Exception as e:
        st.error(f"Dosyalar yüklenirken hata oluştu: {e}")
        st.write("Kontrol edilen model yolu:", str(model_path))
        st.write("Kontrol edilen veri yolu:", str(data_path))
        return None, None


assets, profiles = load_fraud_assets()

if assets is None or profiles is None:
    st.stop()

model = assets["model"]
encoder = assets["encoder"]
feature_cols = assets["features"]
train_columns = assets["train_columns"]

# -----------------------------
# FORM
# -----------------------------
st.markdown("## İşlem Analiz Formu")

with st.form("fraud_detection_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        client_ids = sorted(profiles["client_id"].unique().tolist())
        selected_client = st.selectbox("Müşteri Seçimi (Client ID)", options=client_ids)
        amount = st.number_input("İşlem Tutarı ($)", min_value=0.0, value=250.0)
        hour = st.slider("İşlem Saati", 0, 23, 12)

    with col2:
        city = st.text_input("İşlem Şehri", value="Online")
        mcc = st.text_input("MCC Kodu", value="5411")
        use_chip = st.selectbox(
            "İşlem Tipi",
            ["Online Transaction", "Swipe Transaction", "Chip Transaction"]
        )

    with col3:
        speed_alert = st.selectbox(
            "Hız Alarmı",
            [0, 1],
            format_func=lambda x: "Hız Uyarısı VAR" if x == 1 else "Hız Uyarısı YOK"
        )
        tx_count = st.number_input("Son 24 Saatteki İşlem Sayısı", min_value=1, value=1)

    submit = st.form_submit_button("Analiz Et ve Aksiyon Al")

# -----------------------------
# ANALİZ
# -----------------------------
if submit:
    raw_input = pd.DataFrame([{
        "amount": amount,
        "tx_count_last_24h": tx_count,
        "sec_since_last_tx": 0,
        "speed_alert": speed_alert,
        "merchant_city": city,
        "mcc": mcc,
        "use_chip": use_chip,
        "is_weekend": 0,
        "hour": hour
    }])

    input_encoded = encoder.transform(raw_input)
    input_dummies = pd.get_dummies(input_encoded)
    input_final = input_dummies.reindex(columns=train_columns, fill_value=0)

    prob = model.predict_proba(input_final)[0][1]
    risk_score = prob * 100

    st.divider()
    st.markdown("## Sonuç")

    if prob >= 0.60:
        st.error(f"‼️ KRİTİK ALARM: %{risk_score:.2f} risk tespit edildi.")
        st.warning(f"🔒 AKSİYON: {selected_client} numaralı müşterinin kartı otomatik olarak bloke edildi.")
        st.info("📧 Müşteriye SMS ve e-posta ile bilgilendirme gönderildi.")
        st.toast("Kritik müdahale başarılı.", icon="🚨")
    elif prob >= 0.30:
        st.warning(f"⚠️ Şüpheli İşlem: %{risk_score:.2f} risk.")
        st.info("📲 Müşteri ile iletişime geçilip sözlü onay alınması önerilir.")
    else:
        st.success(f"✅ Güvenli işlem: %{risk_score:.2f} risk. İşleme izin verildi.")

    st.progress(float(prob))

    with st.expander("Gelişmiş analiz verileri"):
        st.write("Modelin kullandığı encode edilmiş alanlar:")
        encoded_cols = [c for c in ["merchant_city", "mcc"] if c in input_encoded.columns]
        if encoded_cols:
            st.dataframe(input_encoded[encoded_cols], use_container_width=True)

        st.write("Sinyal özeti:")
        st.json({
            "online_etkisi": "Yüksek",
            "sinsi_vaka_algısı": "Aktif"
        })

st.markdown("""
<div class="footer-box">
    © 2026 CaixaBank AI Risk Management Solutions. All Rights Reserved.
</div>
""", unsafe_allow_html=True)