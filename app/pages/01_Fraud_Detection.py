import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from pathlib import Path

# --- MODEL YÜKLEME (GÜNCEL YOL) ---
@st.cache_resource
def load_fraud_system():
    # Dosya konumunu akıllıca bul
    current_file_path = Path(__file__).resolve()
    # app/pages/01_Fraud_Detection.py -> bank-shield-ai/models
    base_dir = current_file_path.parent.parent.parent 
    model_path = base_dir / "models" / "fraud_dream_model_bundle.joblib"
    
    if not model_path.exists():
        st.error(f"Model dosyası bulunamadı: {model_path}")
        return None
    return joblib.load(str(model_path))

system = load_fraud_system()

if system:
    model = system['model']
    encoder = system['encoder']
    feature_order = system['features']

    st.title("🛡️ Bank Shield AI: Risk Analiz Sistemi")
    st.markdown("İşlem detaylarını girerek bankacılık güvenliğini kontrol edin.")

    # --- FORM YAPISI ---
    with st.form("fraud_analysis_form"):
        st.subheader("İşlem Bilgileri")
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("İşlem Tutarı ($)", min_value=0.0, value=150.0)
            city = st.text_input("Şehir (Merchant City)", value="Beulah")
            mcc = st.text_input("MCC Kodu", value="5912")
            use_chip = st.selectbox("İşlem Tipi", ["Swipe Transaction", "Chip Transaction", "Online Transaction"])
        with c2:
            hour = st.slider("İşlem Saati", 0, 23, 12)
            is_weekend = st.radio("Haftasonu mu?", [0, 1], format_func=lambda x: "Evet" if x==1 else "Hayır")
            tx_count_24h = st.number_input("Son 24 Saatlik İşlem Sayısı", min_value=1, value=1)
            speed_alert = st.selectbox("Hız Alarmı", [0, 1], format_func=lambda x: "Var" if x==1 else "Yok")

        st.subheader("Müşteri Bilgileri")
        c3, c4 = st.columns(2)
        with c3:
            age = st.slider("Müşteri Yaşı", 18, 100, 35)
            credit_score = st.slider("Kredi Skoru", 300, 850, 650)
        with c4:
            gender = st.selectbox("Cinsiyet", ["Male", "Female"])
            income = st.number_input("Yıllık Gelir ($)", min_value=0, value=50000)

        # Formun kendi butonu
        submit = st.form_submit_button("Risk Analizi Yap")

    # --- ANALİZ MANTIĞI ---
    if submit:
        try:
            with st.spinner("Yapay zeka analiz ediyor..."):
                # 1. Girdiyi DataFrame Yap
                raw_input = pd.DataFrame([{
                    'amount': amount, 'tx_count_last_24h': tx_count_24h,
                    'sec_since_last_tx': 0, 'speed_alert': speed_alert,
                    'merchant_city': city, 'mcc': mcc, 'use_chip': use_chip,
                    'gender': gender, 'current_age': age, 'credit_score': credit_score,
                    'yearly_income': income, 'is_weekend': is_weekend, 'hour': hour, 'total_debt': 0
                }])

                # 2. Encoding işlemleri
                encoded_input = encoder.transform(raw_input)
                final_input = pd.get_dummies(encoded_input)
                
                for col in feature_order:
                    if col not in final_input.columns:
                        final_input[col] = 0
                final_input = final_input[feature_order]

                # 3. Tahmin
                prob = model.predict_proba(final_input)[0][1]
                
                # --- SONUÇ EKRANI ---
                st.divider()
                if prob >= 0.75:
                    st.error(f"🚨 KRİTİK RİSK: %{prob*100:.2f}")
                    st.warning("Kartı bloke etmeniz önerilir.")
                elif prob >= 0.40:
                    st.warning(f"⚠️ ORTA RİSK: %{prob*100:.2f}")
                else:
                    st.success(f"✅ DÜŞÜK RİSK: %{prob*100:.2f}")
                    st.balloons()
                
                st.progress(float(prob))
        
        except Exception as e:
            st.error(f"Analiz sırasında bir hata oluştu: {e}")
else:
    st.warning("Sistem yüklenemediği için analiz yapılamıyor. Lütfen modellerin varlığını kontrol edin.")