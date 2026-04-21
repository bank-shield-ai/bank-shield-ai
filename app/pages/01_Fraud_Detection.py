import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import smtplib # Mail için

# --- MODEL VE VERİ YÜKLEME ---
@st.cache_resource
def load_fraud_assets():
    current_file_path = Path(__file__).resolve()
    base_dir = current_file_path.parent.parent.parent 
    
    # Model paketi
    model_path = base_dir / "models" / "fraud_dream_model_bundle.joblib"
    # Müşteri listesini çekmek için summary datası
    data_path = base_dir / "data" / "processed" / "customer_risk_summary.parquet"
    
    bundle = joblib.load(str(model_path))
    profiles = pd.read_parquet(data_path)
    return bundle, profiles

try:
    assets, profiles = load_fraud_assets()
    model = assets['model']
    encoder = assets['encoder']
    feature_order = assets['features']
except Exception as e:
    st.error("Gerekli dosyalar yüklenemedi. Lütfen yolları kontrol edin.")
    st.stop()

# --- BAŞLIK ---
st.title("🚨 Bank Shield AI: Anlık Müdahale Paneli")
st.markdown("Şüpheli işlemleri tespit edin ve anında bloke edin.")

# --- FORM ---
with st.form("action_fraud_form"):
    st.subheader("Müşteri ve İşlem Seçimi")
    
    # Kimin işlemi olduğunu seçiyoruz
    client_ids = sorted(profiles["client_id"].unique().tolist())
    selected_client = st.selectbox("İşlem Yapan Müşteri (Client ID)", options=client_ids)
    
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("İşlem Tutarı ($)", min_value=0.0, value=250.0)
        city = st.text_input("Merchant City (İşlem Yapılan Şehir)", value="Beulah")
        mcc = st.text_input("MCC Kodu", value="5912")
    
    with col2:
        use_chip = st.selectbox("İşlem Tipi", ["Swipe Transaction", "Chip Transaction", "Online Transaction"])
        hour = st.slider("İşlem Saati", 0, 23, 12)
        speed_alert = st.selectbox("Hız Alarmı (Konum Sapması)", [0, 1], format_func=lambda x: "VAR" if x==1 else "YOK")

    submit = st.form_submit_button("ANALİZ ET VE AKSİYON AL")

# --- ANALİZ VE OTOMASYON ---
if submit:
    # Modelin beklediği ama arayüzden sildiğimiz değerleri 
    # müşterinin profilinden veya varsayılan değerlerden çekiyoruz
    cust_profile = profiles[profiles["client_id"] == selected_client].iloc[0]
    
    raw_input = pd.DataFrame([{
        'amount': amount,
        'tx_count_last_24h': 1,
        'sec_since_last_tx': 0,
        'speed_alert': speed_alert,
        'merchant_city': city,
        'mcc': mcc,
        'use_chip': use_chip,
        'gender': cust_profile.get('gender', 'Male'), # Arka planda kalsın
        'current_age': cust_profile.get('current_age', 35),
        'credit_score': cust_profile.get('credit_score_first', 650),
        'yearly_income': cust_profile.get('yearly_income_first', 50000),
        'is_weekend': 0,
        'hour': hour,
        'total_debt': 0
    }])

    # Tahmin
    encoded_input = encoder.transform(raw_input)
    final_input = pd.get_dummies(encoded_input)
    for col in feature_order:
        if col not in final_input.columns: final_input[col] = 0
    final_input = final_input[feature_order]

    prob = model.predict_proba(final_input)[0][1]

    # --- KRİTİK MÜDAHALE EKRANI ---
    st.divider()
    
    if prob >= 0.70:
        st.error(f"‼️ KRİTİK ALARM: Müşteri {selected_client} için sahtecilik riski: %{prob*100:.1f}")
        
        # OTOMATİK AKSİYONLAR
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"📧 Bilgi: {selected_client} numaralı müşterinin kayıtlı mailine uyarı gönderildi.")
        with c2:
            st.warning(f"🔒 DURUM: Müşteri kartı geçici olarak BLOKE EDİLDİ.")
            
        st.toast(f"Aksiyon Tamamlandı: {selected_client} koruma altında.")
    
    elif prob >= 0.35:
        st.warning(f"⚠️ Şüpheli İşlem: %{prob*100:.1f}. Müşteri {selected_client} aranarak onay alınmalı.")
    else:
        st.success(f"✅ Güvenli İşlem: %{prob*100:.1f}. Müşteri {selected_client} işlemi onaylandı.")

    st.progress(float(prob))