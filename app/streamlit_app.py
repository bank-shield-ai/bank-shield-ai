import streamlit as st
import base64
from pathlib import Path

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="CaixaBank AI Risk Hub", page_icon="🏦", layout="wide")

# --- PATHS (Dosya Yolları) ---
APP_DIR = Path(__file__).resolve().parent
logo_path = APP_DIR / "logo.png"
about_img_path = APP_DIR / "about_image.png"
beha_img_path = APP_DIR / "beha.png"
loan_img_path = APP_DIR / "loan_image.png"

# --- HELPER: Base64 Görsel Okuma ---
def get_base64_image(image_path):
    try:
        with open(str(image_path), "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return None

# --- SESSION & NAVIGATION ---
def init_session():
    if "user_role" not in st.session_state: 
        st.session_state.user_role = None
    if "behavioral_view_mode" not in st.session_state: 
        st.session_state.behavioral_view_mode = "customer"

init_session()

# --- CSS (SENİN BEĞENDİĞİN TASARIMIN AYNISI) ---
st.markdown("""
<style>
    .main { background-color: white !important; }
    .block-container { padding: 1.5rem 4rem !important; max-width: 1500px !important; }

    .header-box { display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; }

    .hero-box {
        background: linear-gradient(135deg, #004587 0%, #002e5a 100%);
        color: white; padding: 85px 55px; border-radius: 35px;
        margin-bottom: 50px; box-shadow: 0 15px 35px rgba(0,69,135,0.1);
    }
    .hero-title { font-size: 3.8rem; font-weight: 800; margin-bottom: 15px; }
    .hero-sub { font-size: 1.4rem; opacity: 0.9; }

    /* Kart Tasarımı */
    .custom-card {
        background: white; border-radius: 28px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06);
        height: 520px; overflow: hidden;
        display: flex; flex-direction: column;
        border: 1px solid #f0f2f5;
    }
    .card-img { width: 100%; height: 280px; object-fit: cover; }
    .card-text-area { padding: 25px; text-align: center; flex-grow: 1; display: flex; flex-direction: column; justify-content: center; }
    .card-title { font-size: 1.6rem; font-weight: 700; color: #1f2d3d; margin-bottom: 12px; }
    .card-desc { color: #667085; font-size: 1rem; line-height: 1.6; font-weight: 500; }

    /* Butonlar */
    div.stButton > button {
        background-color: #004587 !important; color: white !important;
        border-radius: 30px !important; border: none !important;
        padding: 12px 28px !important; font-weight: 600 !important;
        width: 100% !important; min-height: 48px;
    }

    div[data-testid="stPopover"] button {
        background-color: #00a550 !important; border-radius: 12px !important;
    }

    /* TURUNCU BUTON ÖZEL */
    [data-testid="column"]:nth-child(2) div.stButton > button {
        background-color: #ff6600 !important;
    }

    .footer { text-align: center; padding: 50px; margin-top: 60px; border-top: 1px solid #eee; color: #888; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col_logo, col_nav = st.columns([2.5, 1])

with col_logo:
    logo_base64 = get_base64_image(logo_path)
    if logo_base64:
        st.markdown(f'<img src="data:image/png;base64,{logo_base64}" style="width:350px;">', unsafe_allow_html=True)
    else:
        st.header("CaixaBank")

with col_nav:
    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        st.button("Müşteri Ol", key="top_reg")
    with c_btn2:
        with st.popover("Giriş Yap", use_container_width=True):
            if st.button("Yetkili Girişi", key="auth_l"):
                st.session_state.user_role = "authorized"
                st.switch_page("pages/00_Yetkili_Paneli.py")
            if st.button("Müşteri Girişi", key="cust_l"):
                st.session_state.user_role = "customer"
                st.switch_page("pages/02_Behavioral_Risk.py")

# --- HERO ---
st.markdown(f"""
<div class="hero-box">
    <div class="hero-title">Güvenliğiniz, Bizim Önceliğimiz.</div>
    <div class="hero-sub">Yapay zeka destekli Risk Yönetim Portalı ile tüm işlemlerinizi 7/24 kontrol altında tutun.</div>
</div>
""", unsafe_allow_html=True)

# --- ANA İÇERİK ---
col1, col2, col3 = st.columns(3, gap="large")

about_img = get_base64_image(about_img_path)
beha_img = get_base64_image(beha_img_path)
loan_img = get_base64_image(loan_img_path)

# 1. KART
with col1:
    st.markdown(f"""
    <div class="custom-card">
        <img src="data:image/png;base64,{about_img}" class="card-img">
        <div class="card-text-area">
            <div class="card-desc">CaixaBank hakkında daha fazla bilgi alın.</div>
        </div>
    """, unsafe_allow_html=True)
    st.button("Hakkımızda", key="abt_btn", disabled=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 2. KART (TURUNCU BUTON)
with col2:
    st.markdown(f"""
    <div class="custom-card">
        <img src="data:image/png;base64,{beha_img}" class="card-img">
        <div class="card-text-area">
            <div class="card-title">📊 Anlık Verilerle Risk Skorunu Hesapla</div>
        </div>
    """, unsafe_allow_html=True)
    # on_click rerun hatasına sebep olabildiği için if kontrolü ile yönlendirme yaptık
    if st.button("Hemen Başla", key="beh_btn"):
        st.session_state.user_role = "customer"
        st.session_state.behavioral_view_mode = "customer"
        st.switch_page("pages/02_Behavioral_Risk.py")
    st.markdown("</div>", unsafe_allow_html=True)

# 3. KART
with col3:
    st.markdown(f"""
    <div class="custom-card">
        <img src="data:image/png;base64,{loan_img}" class="card-img">
        <div class="card-text-area">
            <div class="card-desc">5000 Dolara kadar faizsiz kredi imkanı</div>
        </div>
    """, unsafe_allow_html=True)
    st.button("Başvur", key="loan_btn")
    st.markdown("</div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div class="footer">
    © 2026 CaixaBank AI Risk Management Solutions. All Rights Reserved.<br>
    <small>Güvenliğiniz için tüm işlemler uçtan uca şifrelenmektedir.</small>
</div>
""", unsafe_allow_html=True)