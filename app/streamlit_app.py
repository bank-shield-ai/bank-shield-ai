import streamlit as st

st.set_page_config(page_title="CaixaBank AI Risk Hub", page_icon="🏦", layout="wide")

# --- GELİŞMİŞ CSS: GERÇEK BANKACILIK TEMASI ---
st.markdown("""
    <style>
    /* Arka plan ve genel font */
    .main { background-color: #f0f2f5; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Üst Header Alanı */
    .header-box {
        background-color: white;
        padding: 20px 50px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 3px solid #ff6600; /* Caixabank Turuncusu */
        margin-bottom: 0px;
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
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    /* Buton Tasarımları */
    div.stButton > button {
        background-color: #004587;
        color: white;
        border-radius: 50px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #ff6600;
        color: white;
        border: none;
    }
    
    /* Sidebar Temizleme */
    section[data-testid="stSidebar"] { background-color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.markdown(f"""
    <div class="header-box">
        <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/CaixaBank.svg" width="180">
        <div style="color: #004587; font-weight: bold;">Kişisel Bankacılık | Kurumsal | AI Risk Portal</div>
    </div>
    """, unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
    <div class="hero-section">
        <h1 style="font-size: 3rem; margin-bottom: 10px; color: white;">Güvenliğiniz, Bizim Önceliğimiz.</h1>
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