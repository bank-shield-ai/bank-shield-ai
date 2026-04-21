import streamlit as st
import base64
from pathlib import Path

# --- SAYFA AYARLARI (En üstte olmalı) ---
st.set_page_config(page_title="CaixaBank AI Risk Hub", page_icon="🏦", layout="wide")

# --- LOGO OKUMA FONKSİYONU ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return None

current_dir = Path(__file__).parent
logo_path = current_dir / "logo.png"
logo_base64 = get_base64_image(str(logo_path))

# --- LOGO HAZIRLAMA ---
if logo_base64:
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="200">'
else:
    logo_html = '<h2 style="color: #004587; margin:0;">CaixaBank</h2>'

# --- GELİŞMİŞ CSS: BANKACILIK TEMASI VE SAĞ ÜST BUTONLAR ---
st.markdown("""
    <style>
    /* Arka plan ve genel font */
    .main { background-color: #f0f2f5; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Üst Header Alanı */
    .header-box {
        background-color: white;
        padding: 15px 50px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 3px solid #ff6600;
        margin-bottom: 0px;
    }

    /* Sağ Üst Buton Menüsü */
    .header-right-menu {
        display: flex;
        gap: 15px;
        align-items: center;
    }
    
    /* Müşteri Ol Butonu (Kırmızı) */
    .btn-register {
        background-color: #eb0000;
        color: white !important;
        padding: 10px 25px;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
        transition: 0.3s;
        font-size: 14px;
    }
    
    /* Giriş Yap Butonu (Yeşil) */
    .btn-login {
        background-color: #00a550;
        color: white !important;
        padding: 10px 25px;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
        transition: 0.3s;
        font-size: 14px;
    }

    .btn-register:hover, .btn-login:hover {
        opacity: 0.8;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        color: white !important;
    }
    
    /* Hero Section (Giriş Bannerı) */
    .hero-section {
        background: linear-gradient(135deg, #004587 0%, #002e5a 100%);
        color: white;
        padding: 60px 50px;
        text-align: left;
        border-radius: 0 0 50px 50px;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Kart Tasarımları */
    .service-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
        border: 1px solid #e1e4e8;
        height: 100%;
    }
    .service-card:hover {
        transform: translateY(-10px);
    }
    
    /* Streamlit Butonlarını Özelleştirme */
    div.stButton > button {
        background-color: #004587;
        color: white;
        border-radius: 50px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #ff6600;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# --- HEADER (MENÜ + BUTONLAR KOMBO) ---
st.markdown(f"""
    <div class="header-box">
        <div style="display: flex; align-items: center;">
            {logo_html}
        </div>
        
        <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 8px;">
            <div style="color: #004587; font-size: 13px; font-weight: 500; letter-spacing: 0.5px;">
                Bireysel &nbsp; | &nbsp; Kurumsal &nbsp; | &nbsp; <span style="font-weight: bold; color: #ff6600;">AI Risk Portal</span>
            </div>
            
            <div class="header-right-menu">
                <a href="#" class="btn-register">Müşteri Ol</a>
                <a href="#" class="btn-login">Giriş Yap</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
    <div class="hero-section">
        <h1 style="font-size: 3rem; margin-bottom: 10px; color: white; border:none;">Güvenliğiniz, Bizim Önceliğimiz.</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Yapay zeka destekli Risk Yönetim Portalı ile tüm işlemlerinizi 7/24 kontrol altında tutun.</p>
    </div>
    """, unsafe_allow_html=True)

# --- SERVİS KARTLARI ---
st.markdown('<div style="padding: 0 50px;">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("""
        <div class="service-card">
            <h2 style="color: #004587;">🔍 Fraud Detection</h2>
            <p style="color: #666;">Şüpheli işlemlerinizi anlık olarak XGBoost motorumuzla analiz edin ve sahteciliği saniyeler içinde durdurun.</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Hemen Analiz Et ➔", key="btn_fraud"):
        st.switch_page("pages/01_Fraud_Detection.py")

with col2:
    st.markdown("""
        <div class="service-card">
            <h2 style="color: #004587;">📊 Behavioral Risk</h2>
            <p style="color: #666;">Müşteri davranışlarını ve geçmiş verileri inceleyerek uzun vadeli risk skorlarını görüntüleyin.</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Skorları Görüntüle ➔", key="btn_risk"):
        st.switch_page("pages/02_Behavioral_Risk.py")

with col3:
    st.markdown("""
        <div class="service-card">
            <h2 style="color: #004587;">🛡️ Security Logs</h2>
            <p style="color: #666;">Sistem üzerindeki geçmiş alarmları ve müdahale raporlarını detaylı olarak inceleyin.</p>
        </div>
    """, unsafe_allow_html=True)
    st.button("Raporlar (Yakında)", disabled=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- ALT BİLGİ (FOOTER) ---
st.markdown("""
    <div style="text-align: center; margin-top: 100px; padding: 20px; color: #999; font-size: 0.8rem;">
        © 2026 CaixaBank AI Risk Management Solutions. All Rights Reserved.
    </div>
    """, unsafe_allow_html=True)