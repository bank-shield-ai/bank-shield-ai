import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="BankShield AI - Behavioral Risk",
    page_icon="🛡️",
    layout="centered"
)

# Notebook'tan çıkan beklenen feature listesi
DEFAULT_FEATURES = [
    "amount_mean",
    "amount_std",
    "amount_max",
    "amount_sum",
    "is_night_transaction_mean",
    "fast_tx_mean",
    "yearly_income_first",
    "total_debt_first",
    "credit_score_first",
]

MODEL_CANDIDATES = [
    "behavioral_risk_model.joblib",
    "mvp_behavioral_risk.joblib",
    "behavioral_risk_model.pkl",
    "fraud_model.pkl",
]

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


@st.cache_resource
def load_model_payload():
    """
    Model payload'ını yükler.
    Öncelik sırası:
    1) behavioral_risk_model.joblib
    2) mvp_behavioral_risk.joblib
    3) diğer adaylar
    """
    for filename in MODEL_CANDIDATES:
        model_path = MODELS_DIR / filename
        if model_path.exists():
            loaded = joblib.load(model_path)

            # Eğer payload dict ise
            if isinstance(loaded, dict):
                model = loaded.get("model", loaded)
                features = loaded.get("features", DEFAULT_FEATURES)
                threshold = loaded.get("threshold", None)
            else:
                # Sadece model objesi kaydedildiyse
                model = loaded
                features = DEFAULT_FEATURES
                threshold = None

            return {
                "model": model,
                "features": features,
                "threshold": threshold,
                "model_path": str(model_path)
            }

    raise FileNotFoundError(
        f"Models klasöründe uygun model bulunamadı. Kontrol edilen dosyalar: {MODEL_CANDIDATES}"
    )


def risk_group(score: int) -> str:
    if score < 30:
        return "DÜŞÜK (Güvenli)"
    elif score < 70:
        return "ORTA (Takip)"
    return "YÜKSEK (Kritik)"


def recommendation(score: int) -> str:
    if score < 30:
        return "Müşteri düşük riskli görünüyor. Standart izleme yeterli."
    elif score < 70:
        return "Müşteri orta risk bandında. Ek kontrol ve periyodik takip önerilir."
    return "Müşteri yüksek riskli görünüyor. Manuel inceleme ve sıkı doğrulama önerilir."


def build_input_df(feature_order, values_dict):
    row = {}
    for feature in feature_order:
        row[feature] = values_dict.get(feature, 0)
    return pd.DataFrame([row])


st.title("🛡️ BankShield AI")
st.subheader("Behavioral Risk Scoring Demo")

st.markdown(
    """
Bu ekran müşteri davranışsal profiline göre **0-100 arası risk puanı** üretir.
"""
)

try:
    payload = load_model_payload()
    model = payload["model"]
    feature_order = payload["features"]
    threshold = payload["threshold"]
    loaded_model_path = payload["model_path"]

    st.success(f"Model başarıyla yüklendi: `{Path(loaded_model_path).name}`")

except Exception as e:
    st.error("Model yüklenemedi.")
    st.code(str(e))
    st.stop()


with st.expander("Modelin beklediği feature sırası"):
    st.write(feature_order)
    if threshold is not None:
        st.write(f"Eğitim sırasında kullanılan risk eşiği bilgisi mevcut: {threshold:.4f}")

st.markdown("---")
st.markdown("### Müşteri Girdi Formu")

with st.form("behavioral_risk_form"):
    amount_mean = st.number_input("Ortalama İşlem Tutarı (amount_mean)", min_value=0.0, value=75.0, step=1.0)
    amount_std = st.number_input("İşlem Sapması (amount_std)", min_value=0.0, value=25.0, step=1.0)
    amount_max = st.number_input("Maksimum İşlem Tutarı (amount_max)", min_value=0.0, value=300.0, step=1.0)
    amount_sum = st.number_input("Toplam İşlem Tutarı (amount_sum)", min_value=0.0, value=5000.0, step=10.0)

    is_night_transaction_mean = st.slider(
        "Gece İşlem Oranı (is_night_transaction_mean)",
        min_value=0.0, max_value=1.0, value=0.05, step=0.01
    )

    fast_tx_mean = st.slider(
        "Hızlı İşlem Oranı (fast_tx_mean)",
        min_value=0.0, max_value=1.0, value=0.02, step=0.01
    )

    yearly_income_first = st.number_input("Yıllık Gelir (yearly_income_first)", min_value=0.0, value=50000.0, step=1000.0)
    total_debt_first = st.number_input("Toplam Borç (total_debt_first)", min_value=0.0, value=25000.0, step=1000.0)
    credit_score_first = st.number_input("Kredi Skoru (credit_score_first)", min_value=0, max_value=1000, value=650, step=1)

    submitted = st.form_submit_button("Risk Hesapla")


if submitted:
    input_values = {
        "amount_mean": float(amount_mean),
        "amount_std": float(amount_std),
        "amount_max": float(amount_max),
        "amount_sum": float(amount_sum),
        "is_night_transaction_mean": float(is_night_transaction_mean),
        "fast_tx_mean": float(fast_tx_mean),
        "yearly_income_first": float(yearly_income_first),
        "total_debt_first": float(total_debt_first),
        "credit_score_first": float(credit_score_first),
    }

    X_input = build_input_df(feature_order, input_values)

    try:
        prediction = int(model.predict(X_input)[0])

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X_input)[0][1])
        else:
            # predict_proba yoksa güvenli fallback
            probability = float(prediction)

        score = int(round(probability * 100))
        group = risk_group(score)
        advice = recommendation(score)

        st.markdown("---")
        st.markdown("## Sonuç")

        c1, c2 = st.columns(2)
        c1.metric("Risk Puanı", f"{score}/100")
        c2.metric("Tahmin", "Yüksek Risk" if prediction == 1 else "Normal Risk")

        if score < 30:
            st.success(f"Risk Grubu: {group}")
        elif score < 70:
            st.warning(f"Risk Grubu: {group}")
        else:
            st.error(f"Risk Grubu: {group}")

        st.write(advice)

        st.markdown("### Gönderilen Model Girdisi")
        st.dataframe(X_input, use_container_width=True)

    except Exception as e:
        st.error("Tahmin sırasında hata oluştu.")
        st.code(str(e))