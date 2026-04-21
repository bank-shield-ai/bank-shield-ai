import streamlit as st

st.set_page_config(page_title="CaixaBank Risk Hub", page_icon="🏦", layout="wide")

# CaixaBank Renkleri & Stil
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    div.stButton > button {
        background-color: #004587; color: white; border-radius: 5px;
        border: 1px solid #ff6600; height: 3em; width: 100%;
    }
    div.stButton > button:hover { background-color: #ff6600; color: white; }
    h1 { color: #004587; }
    </style>
    """, unsafe_allow_html=True)

st.image("https://upload.wikimedia.org/wikipedia/commons/4/4e/CaixaBank.svg", width=250)
st.title("🛡️ Risk Management Control Center")
st.markdown("### Hoş geldiniz. Lütfen analiz yapmak istediğiniz modülü seçin.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Fraud Detection")
    st.write("Anlık işlem bazlı sahtecilik analizi ve XGBoost risk skorlaması.")
    if st.button("Fraud Paneline Git"):
        st.switch_page("pages/01_Fraud_Detection.py")

with col2:
    st.subheader("📊 Behavioral Risk")
    st.write("Müşteri geçmişi ve davranışsal profil risk analizi.")
    if st.button("Müşteri Skoruna Git"):
        st.switch_page("pages/02_Behavioral_Risk.py")