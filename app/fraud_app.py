import streamlit as st
import pandas as pd
import joblib
import traceback
from pathlib import Path

from feature_builder import build_features

st.set_page_config(
    page_title="BankShield AI - Fraud Tespiti",
    page_icon="🚨",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fraud_model_final_v5.joblib"
CLIENT_SUMMARY_PATH = BASE_DIR / "data" / "processed" / "client_history_summary.parquet"
MERCHANT_RISK_PATH = BASE_DIR / "data" / "processed" / "merchant_risk_lookup.parquet"

CHALLENGE_THRESHOLD = 0.50
DECLINE_THRESHOLD = 0.80


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    if not CLIENT_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Client summary bulunamadı: {CLIENT_SUMMARY_PATH}")
    if not MERCHANT_RISK_PATH.exists():
        raise FileNotFoundError(f"Merchant risk lookup bulunamadı: {MERCHANT_RISK_PATH}")

    client_summary_df = pd.read_parquet(CLIENT_SUMMARY_PATH)
    merchant_risk_df = pd.read_parquet(MERCHANT_RISK_PATH)

    return client_summary_df, merchant_risk_df


def get_decision(prob: float) -> str:
    if prob >= DECLINE_THRESHOLD:
        return "DECLINE"
    elif prob >= CHALLENGE_THRESHOLD:
        return "CHALLENGE"
    return "PASS"


def get_risk_band(prob: float) -> str:
    if prob >= DECLINE_THRESHOLD:
        return "Yüksek Risk"
    elif prob >= CHALLENGE_THRESHOLD:
        return "Orta Risk"
    return "Düşük Risk"


st.title("🚨 Fraud Tespiti")
st.caption("İşlem bilgileri manuel girilir, davranışsal metrikler geçmiş veriden otomatik hesaplanır.")

try:
    model = load_model()
    client_summary_df, merchant_risk_df = load_reference_data()
    st.success("Model ve referans veriler başarıyla yüklendi.")
except Exception as e:
    st.error("Yükleme hatası oluştu.")
    st.code(str(e))
    st.code(traceback.format_exc())
    st.stop()

st.markdown("## İşlem Girdi Alanları")

sol, sag = st.columns(2)

with sol:
    client_id = st.text_input("Client ID", "")
    amount = st.number_input("İşlem Tutarı", min_value=0.0, value=50.0, step=1.0)

    islem_tipi = st.selectbox(
        "İşlem Tipi",
        ["Online Transaction", "Chip Transaction", "Swipe Transaction"]
    )

    merchant_city = st.text_input("İşlem Şehri", "ONLINE")
    merchant_state = st.text_input("İşlem Eyaleti / Bölgesi", "ONLINE")
    mcc = st.text_input("MCC Kodu", "5411")
    is_return = st.selectbox("İade İşlemi mi?", [0, 1])

with sag:
    hour = st.slider("İşlem Saati", 0, 23, 12)
    day = st.slider("Gün", 1, 31, 15)
    month = st.slider("Ay", 1, 12, 6)
    year = st.slider("Yıl", 2010, 2025, 2019)

    haftanin_gunu = st.selectbox(
        "Haftanın Günü",
        ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
    )

st.info(
    f"Karar politikası: PASS < {CHALLENGE_THRESHOLD:.2f} | "
    f"CHALLENGE {CHALLENGE_THRESHOLD:.2f}–{DECLINE_THRESHOLD:.2f} | "
    f"DECLINE ≥ {DECLINE_THRESHOLD:.2f}"
)

if st.button("Fraud Tahmini Yap", type="primary"):
    try:
        if not client_id.strip():
            st.warning("Lütfen client_id girin.")
            st.stop()

        known_clients = set(client_summary_df["client_id"].astype(str).str.strip())
        if str(client_id).strip() not in known_clients:
            st.warning("Bu client_id geçmiş özet tabloda bulunamadı. Global fallback profil kullanılacak.")

        tx_datetime = pd.Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour
        )

        df_input = build_features(
            client_id=client_id.strip(),
            amount=amount,
            islem_tipi=islem_tipi,
            hour=hour,
            day=day,
            month=month,
            year=year,
            haftanin_gunu_tr=haftanin_gunu,
            merchant_city=merchant_city,
            merchant_state=merchant_state,
            mcc=mcc,
            is_return=is_return,
            tx_datetime=tx_datetime,
            client_summary_df=client_summary_df,
            merchant_risk_df=merchant_risk_df
        )

        fraud_prob = float(model.predict_proba(df_input)[0, 1])
        risk_score = round(fraud_prob * 100, 2)
        decision = get_decision(fraud_prob)
        band = get_risk_band(fraud_prob)

        st.markdown("## Sonuç")

        r1, r2, r3 = st.columns(3)
        r1.metric("Fraud Olasılığı", f"{fraud_prob:.2%}")
        r2.metric("Risk Skoru", f"{risk_score:.2f}/100")
        r3.metric("Karar", decision)

        if decision == "DECLINE":
            st.error("Yüksek Riskli İşlem — DECLINE 🚨")
        elif decision == "CHALLENGE":
            st.warning("Orta Riskli İşlem — CHALLENGE ⚠️")
        else:
            st.success("Düşük Riskli İşlem — PASS ✅")

        st.write(f"**Risk Bandı:** {band}")

        if decision == "DECLINE":
            st.error(
                f"Bu işlem yüksek confidence risk grubunda. "
                f"({fraud_prob:.4f} ≥ {DECLINE_THRESHOLD:.2f})"
            )
        elif decision == "CHALLENGE":
            st.warning(
                f"Bu işlem orta risk grubunda. "
                f"({CHALLENGE_THRESHOLD:.2f} ≤ {fraud_prob:.4f} < {DECLINE_THRESHOLD:.2f})"
            )
        else:
            st.info(
                f"Bu işlem düşük risk grubunda. "
                f"({fraud_prob:.4f} < {CHALLENGE_THRESHOLD:.2f})"
            )

        st.markdown("## Hesaplanan Feature'lar")

        show_cols = [
            "amount",
            "log_amount",
            "amount_zscore",
            "amount_to_limit_ratio",
            "user_tx_count",
            "user_mean_amount",
            "user_std_amount",
            "amount_deviation",
            "time_diff",
            "fast_tx",
            "very_fast_tx",
            "rolling_mean_3",
            "rolling_std_3",
            "rolling_amount_deviation",
            "amount_to_user_mean_ratio",
            "amount_to_rolling_mean_ratio",
            "abs_amount_deviation",
            "abs_rolling_amount_deviation",
            "tx_velocity",
            "merchant_risk_score",
            "merchant_risk_score_log",
            "current_age",
            "yearly_income",
            "total_debt",
            "credit_score",
            "gender",
            "per_capita_income",
            "amount_spike_2std",
            "amount_spike_3std",
            "log_amount_to_income",
            "debt_to_income_ratio",
            "mcc",
            "is_return",
            "hour",
            "day",
            "month",
            "year",
            "day_of_week",
            "day_of_week_num",
            "is_weekend",
            "is_night",
            "is_peak_hour",
        ]

        existing_show_cols = [col for col in show_cols if col in df_input.columns]
        st.dataframe(df_input[existing_show_cols], use_container_width=True)

        with st.expander("🧠 Debug - Model Bilgisi", expanded=False):
            st.write("Model path:", str(MODEL_PATH))
            st.write("Client summary path:", str(CLIENT_SUMMARY_PATH))
            st.write("Merchant risk path:", str(MERCHANT_RISK_PATH))
            st.write("Model type:", str(type(model)))

        with st.expander("📦 Modele Gönderilen Input", expanded=False):
            st.dataframe(df_input, use_container_width=True)

    except Exception as e:
        st.error("Tahmin sırasında hata oluştu.")
        st.code(str(e))
        st.code(traceback.format_exc())