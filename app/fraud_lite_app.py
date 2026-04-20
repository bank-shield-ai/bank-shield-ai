import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="BankShield AI - Fraud Lite",
    page_icon="🚨",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_lite_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "fraud_lite_feature_columns.joblib"
CATEGORIES_PATH = BASE_DIR / "models" / "fraud_lite_category_values.joblib"


@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature columns bulunamadı: {FEATURES_PATH}")
    if not CATEGORIES_PATH.exists():
        raise FileNotFoundError(f"Category values bulunamadı: {CATEGORIES_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    category_values = joblib.load(CATEGORIES_PATH)

    return model, feature_columns, category_values


def risk_band(prob: float) -> str:
    if prob >= 0.80:
        return "High Risk"
    elif prob >= 0.50:
        return "Medium Risk"
    return "Low Risk"


def build_input_dataframe(
    amount,
    use_chip,
    merchant_city,
    mcc,
    hour,
    day,
    month,
    year,
    very_fast_tx,
    day_of_week,
):
    return pd.DataFrame([{
        "amount": amount,
        "use_chip": use_chip,
        "merchant_city": merchant_city,
        "mcc": str(mcc),
        "hour": hour,
        "day": day,
        "month": month,
        "year": year,
        "very_fast_tx": very_fast_tx,
        "day_of_week": day_of_week,
    }])


def preprocess_input(input_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    input_df = input_df.copy()

    # Eğitim tarafındaki mantıkla uyumlu olsun diye kategorikleri dummy'e çevir
    encoded = pd.get_dummies(
        input_df,
        columns=["use_chip", "merchant_city", "day_of_week", "mcc"],
        drop_first=True
    )

    # Eğitimdeki kolon setine hizala
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    return encoded


st.title("🚨 Fraud Detection Lite")
st.caption("Transaction-level fraud probability scoring")

try:
    model, feature_columns, category_values = load_assets()
    st.success("Lite fraud modeli başarıyla yüklendi.")
except Exception as e:
    st.error("Model dosyaları yüklenemedi.")
    st.code(str(e))
    st.stop()

use_chip_values = category_values.get(
    "use_chip_values",
    ["Swipe Transaction", "Chip Transaction", "Online Transaction"]
)
merchant_city_values = category_values.get("merchant_city_values", ["Other"])
day_of_week_values = category_values.get(
    "day_of_week_values",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

st.markdown("## Transaction Inputs")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Amount", min_value=0.0, value=50.0, step=1.0)

    default_use_chip = use_chip_values[0] if use_chip_values else "Swipe Transaction"
    use_chip = st.selectbox(
        "Transaction Type",
        options=use_chip_values,
        index=use_chip_values.index(default_use_chip) if default_use_chip in use_chip_values else 0
    )

    default_city = "Other" if "Other" in merchant_city_values else merchant_city_values[0]
    merchant_city = st.selectbox(
        "Merchant City",
        options=merchant_city_values,
        index=merchant_city_values.index(default_city) if default_city in merchant_city_values else 0
    )

    mcc = st.text_input("MCC", value="5411")

    day_of_week = st.selectbox(
        "Day of Week",
        options=day_of_week_values,
        index=day_of_week_values.index("Wednesday") if "Wednesday" in day_of_week_values else 0
    )

with col2:
    hour = st.slider("Hour", 0, 23, 12)
    day = st.slider("Day", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)
    year = st.slider("Year", 2010, 2025, 2019)
    very_fast_tx = st.selectbox("Very Fast Transaction?", [0, 1], index=0)

st.info(
    "Bu lite model, sade ve sunum dostu bir transaction fraud ekranı için tasarlanmıştır. "
    "Arka planda input verisi dummy encoding ile eğitimdeki kolon yapısına hizalanır."
)

threshold = st.slider(
    "Fraud Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.70,
    step=0.01
)

if st.button("Predict Fraud", type="primary"):
    try:
        input_df = build_input_dataframe(
            amount=amount,
            use_chip=use_chip,
            merchant_city=merchant_city,
            mcc=mcc,
            hour=hour,
            day=day,
            month=month,
            year=year,
            very_fast_tx=very_fast_tx,
            day_of_week=day_of_week,
        )

        encoded_input = preprocess_input(input_df, feature_columns)

        prob = float(model.predict_proba(encoded_input)[0][1])
        pred = int(prob >= threshold)
        score = round(prob * 100, 2)
        band = risk_band(prob)

        st.markdown("## Result")

        r1, r2, r3 = st.columns(3)
        r1.metric("Fraud Probability", f"{prob:.2%}")
        r2.metric("Risk Score", f"{score:.2f}/100")
        r3.metric("Prediction", "Fraud" if pred == 1 else "Not Fraud")

        if band == "High Risk":
            st.error("High Risk 🚨")
        elif band == "Medium Risk":
            st.warning("Medium Risk ⚠️")
        else:
            st.success("Low Risk ✅")

        st.markdown("### Decision Summary")
        st.write(f"**Threshold:** {threshold:.2f}")
        st.write(f"**Risk Band:** {band}")

        with st.expander("Transaction Input Preview"):
            st.dataframe(input_df, use_container_width=True)

        with st.expander("Encoded Model Input"):
            st.write(f"Modele giden kolon sayısı: {encoded_input.shape[1]}")
            st.dataframe(encoded_input, use_container_width=True)

    except Exception as e:
        st.error("Tahmin sırasında hata oluştu.")
        st.code(str(e))