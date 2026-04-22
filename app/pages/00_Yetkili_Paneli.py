import streamlit as st
import base64
from pathlib import Path

st.set_page_config(
    page_title="CaixaBank AI Risk Hub - Yetkili Paneli",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# SESSION
# -----------------------------
if "user_role" not in st.session_state:
    st.session_state.user_role = "authorized"

if "behavioral_view_mode" not in st.session_state:
    st.session_state.behavioral_view_mode = "authorized"

st.session_state.user_role = "authorized"
st.session_state.behavioral_view_mode = "authorized"

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

.nav-links {
    color: #004587;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 6px;
    text-align: right;
}

.hero-section {
    background: linear-gradient(135deg, #004587 0%, #002e5a 100%);
    color: white;
    padding: 48px 40px;
    border-radius: 0 0 36px 36px;
    margin-bottom: 34px;
    box-shadow: 0 10px 28px rgba(0, 69, 135, 0.18);
}

.service-card {
    background: white;
    padding: 28px;
    border-radius: 22px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    min-height: 250px;
    border: 1px solid #e7ebf0;
}

.service-card h3 {
    margin-bottom: 10px;
    color: #1c2b39;
}

.service-card p, .service-card li {
    color: #667085;
    font-size: 14px;
}

div.stButton > button {
    background-color: #004587;
    color: white;
    border-radius: 999px;
    border: none;
    width: 100%;
    font-weight: 700;
    min-height: 42px;
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
    <div>
        <div class="nav-links">
            Yetkili Paneli &nbsp; | &nbsp;
            <span style="font-weight: 700; color: #ff6600;">AI Risk Portal</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# HERO
# -----------------------------
st.markdown("""
<div class="hero-section">
    <h1 style="color: white; margin-bottom: 10px;">Yetkili Kontrol Merkezi</h1>
    <p style="opacity: 0.92; margin-bottom: 0;">
        Sahtecilik analizi ve davranışsal risk modüllerine tek ekrandan erişin.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# KARTLAR
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="service-card">
        <h3>🔍 Fraud Detection</h3>
        <p>Anlık işlem bazlı sahtecilik analizi ve aksiyon motoru.</p>
        <ul>
            <li>Gerçek zamanlı risk skoru</li>
            <li>Kritik alarm üretimi</li>
            <li>Otomatik bloke senaryosu</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Fraud Detection Modülüne Git ➜", key="go_fraud"):
        st.switch_page("pages/01_Fraud_Detection.py")

with col2:
    st.markdown("""
    <div class="service-card">
        <h3>📊 Risk Skoru</h3>
        <p>Müşteri davranış profiline göre risk skorlama ve Ya Şöyle Olsa? simülasyonu.</p>
        <ul>
            <li>Risk skoru hesaplama</li>
            <li>Senaryo testi</li>
            <li>Manuel değişken simülasyonu</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Risk Skoru Modülüne Git ➜", key="go_behavioral_admin"):
        st.session_state.behavioral_view_mode = "authorized"
        st.switch_page("pages/02_Behavioral_Risk.py")

st.markdown("""
<div class="footer-box">
    © 2026 CaixaBank AI Risk Management Solutions. All Rights Reserved.
</div>
""", unsafe_allow_html=True)