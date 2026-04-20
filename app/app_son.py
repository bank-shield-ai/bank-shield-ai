import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Bank Shield AI - Fraud Risk Panel", layout="centered")

# --- MODEL VE ENCODER YÜKLEME ---
@st.cache_resource
def load_fraud_system():
    # 3. Notebook'ta kaydettiğimiz paketi okuyoruz
    bundle = joblib.load("../models/fraud_dream_model_bundle.joblib")
    return bundle

system = load_fraud_system()
model = system['model']
encoder = system['encoder']
feature_order = system['features']

# --- ARAYÜZ (UI) ---
st.title("🛡️ Bank Shield AI: Risk Analiz Sistemi")
st.markdown("İşlem detaylarını girerek bankacılık güvenliğini kontrol edin.")

with st.form("fraud_form"):
    st.subheader("İşlem Bilgileri")
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("İşlem Tutarı ($)", min_value=0.0, value=50.0)
        city = st.text_input("Şehir (Merchant City)", value="Beulah")
        mcc = st.text_input("MCC Kodu", value="5912")
        use_chip = st.selectbox("İşlem Tipi", ["Swipe Transaction", "Chip Transaction", "Online Transaction"])

    with col2:
        hour = st.slider("İşlem Saati", 0, 23, 12)
        is_weekend = st.radio("Haftasonu mu?", [0, 1], format_func=lambda x: "Evet" if x==1 else "Hayır")
        tx_count_24h = st.number_input("Son 24 Saatlik İşlem Sayısı", min_value=1, value=1)
        speed_alert = st.selectbox("Hız Alarmı (Şüpheli Şehir Değişimi)", [0, 1], format_func=lambda x: "Var" if x==1 else "Yok")

    st.subheader("Müşteri Bilgileri")
    col3, col4 = st.columns(2)
    with col3:
        age = st.slider("Müşteri Yaşı", 18, 100, 35)
        credit_score = st.slider("Kredi Skoru", 300, 850, 650)
    with col4:
        gender = st.selectbox("Cinsiyet", ["Male", "Female"])
        income = st.number_input("Yıllık Gelir ($)", min_value=0, value=50000)

    submit = st.form_submit_button("Risk Analizi Yap")

# --- ANALİZ MANTIĞI ---
if submit:
    # 1. Girdiyi DataFrame Yap
    raw_input = pd.DataFrame([{
        'amount': amount,
        'tx_count_last_24h': tx_count_24h,
        'sec_since_last_tx': 0, # Varsayılan
        'speed_alert': speed_alert,
        'merchant_city': city,
        'mcc': mcc,
        'use_chip': use_chip,
        'gender': gender,
        'current_age': age,
        'credit_score': credit_score,
        'yearly_income': income,
        'is_weekend': is_weekend,
        'hour': hour,
        'total_debt': 0 # Gerekirse eklenebilir
    }])

    # 2. Target Encoding (Şehir ve MCC'yi sayıya çevir)
    encoded_input = encoder.transform(raw_input)

    # 3. One-Hot Encoding (use_chip ve gender)
    # Modelin beklediği sütun yapısına getirmek için dummies oluşturuyoruz
    final_input = pd.get_dummies(encoded_input)
    
    # Modelin beklediği tüm sütunları (eksikleri 0 yaparak) tamamla
    for col in feature_order:
        if col not in final_input.columns:
            final_input[col] = 0
            
    final_input = final_input[feature_order] # Sütun sırasını hizala

    # 4. Tahmin ve Olasılık
    prob = model.predict_proba(final_input)[0][1]
    
    # --- RİSK SEVİYESİ BELİRLEME ---
    st.divider()
    if prob >= 0.75:
        st.error(f"🚨 KRİTİK RİSK: %{prob*100:.2f}")
        st.warning("Öneri: Kartı hemen bloke edin ve müşteriyle iletişime geçin.")
    elif prob >= 0.40:
        st.warning(f"⚠️ ORTA RİSK: %{prob*100:.2f}")
        st.info("Öneri: İşlemi incelemeye alın, onay bekletin.")
    else:
        st.success(f"✅ DÜŞÜK RİSK: %{prob*100:.2f}")
        st.balloons()

    # Görsel Risk Barı
    st.progress(float(prob))