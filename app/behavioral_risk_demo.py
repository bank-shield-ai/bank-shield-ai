import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="BankShield AI - Behavioral Risk",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

POSSIBLE_DATA_PATHS = [
    BASE_DIR / "data" / "processed" / "df_combined.parquet",
    BASE_DIR / "notebooks" / "df_combined.parquet",
]

MODEL_CANDIDATES = [
    "behavioral_risk_regressor.joblib",
    "behavioral_risk_model.joblib",
    "mvp_behavioral_risk.joblib",
]

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


def clean_to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce"
    ).fillna(0)


def risk_group(score: float) -> str:
    if score < 30:
        return "DÜŞÜK (Güvenli)"
    elif score < 70:
        return "ORTA (Takip)"
    return "YÜKSEK (Kritik)"


def recommendation(score: float) -> str:
    if score < 30:
        return "Standart izleme yeterli."
    elif score < 70:
        return "Ek kontrol ve periyodik takip önerilir."
    return "Manuel inceleme ve sıkı doğrulama önerilir."


@st.cache_resource
def load_model_payload():
    for filename in MODEL_CANDIDATES:
        model_path = MODELS_DIR / filename
        if model_path.exists():
            loaded = joblib.load(model_path)

            if isinstance(loaded, dict):
                model = loaded.get("model", loaded)
                features = loaded.get("features", DEFAULT_FEATURES)
                model_type = loaded.get("model_type", "classifier")
            else:
                model = loaded
                features = DEFAULT_FEATURES
                model_type = "classifier"

            return {
                "model": model,
                "features": features,
                "model_type": model_type,
                "model_path": str(model_path)
            }

    raise FileNotFoundError(f"Model bulunamadı. Kontrol edilen dosyalar: {MODEL_CANDIDATES}")


def get_data_path():
    for p in POSSIBLE_DATA_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("df_combined.parquet bulunamadı.")


@st.cache_data
def load_raw_data(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "hour" not in df.columns and "date" in df.columns:
        df["hour"] = df["date"].dt.hour

    for col in ["amount", "yearly_income", "total_debt"]:
        if col in df.columns:
            df[col] = clean_to_float(df[col])
        else:
            df[col] = 0

    if "credit_score" not in df.columns:
        df["credit_score"] = 0

    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0

    if "client_id" not in df.columns:
        raise ValueError("Veri setinde client_id kolonu yok.")

    if "hour" in df.columns:
        df["is_night_transaction"] = df["hour"].apply(
            lambda x: 1 if pd.notnull(x) and 0 <= x <= 6 else 0
        )
    else:
        df["is_night_transaction"] = 0

    if "date" in df.columns:
        df = df.sort_values(["client_id", "date"])
        df["fast_tx"] = (
            df.groupby("client_id")["date"]
            .diff()
            .dt.total_seconds()
            .lt(10)
            .fillna(False)
            .astype(int)
        )
    else:
        df["fast_tx"] = 0

    return df


@st.cache_data
def build_customer_profiles(parquet_path: str) -> pd.DataFrame:
    df = load_raw_data(parquet_path)

    customer_df = df.groupby("client_id").agg({
        "amount": ["mean", "std", "max", "sum"],
        "is_fraud": "mean",
        "is_night_transaction": "mean",
        "fast_tx": "mean",
        "yearly_income": "first",
        "total_debt": "first",
        "credit_score": "first",
    })

    customer_df.columns = ["_".join(col).strip() for col in customer_df.columns.values]
    customer_df = customer_df.reset_index()

    customer_df["amount_std"] = customer_df["amount_std"].fillna(0)
    customer_df["yearly_income_first"] = customer_df["yearly_income_first"].replace(0, 1)
    customer_df["debt_to_income"] = (
        customer_df["total_debt_first"] / customer_df["yearly_income_first"]
    )

    return customer_df


def build_model_input(profile_row: pd.Series, feature_order: list) -> pd.DataFrame:
    row = {}
    for feat in feature_order:
        row[feat] = profile_row.get(feat, 0)
    return pd.DataFrame([row])


def score_customer(model, model_type: str, X_input: pd.DataFrame):
    if model_type == "regressor":
        raw_score = float(model.predict(X_input)[0])
        score = max(0, min(100, round(raw_score, 1)))
        pred = 1 if score >= 70 else 0
    else:
        pred = int(model.predict(X_input)[0])
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_input)[0][1])
            score = round(proba * 100, 1)
        else:
            score = float(pred) * 100

    group = risk_group(score)
    advice = recommendation(score)
    return pred, score, group, advice


st.title("🛡️ BankShield AI")
st.caption("Behavioral Risk Scoring Demo")

try:
    payload = load_model_payload()
    model = payload["model"]
    feature_order = payload["features"]
    model_type = payload["model_type"]
    st.success(f"Model yüklendi: {Path(payload['model_path']).name}")
except Exception as e:
    st.error("Model yüklenemedi.")
    st.code(str(e))
    st.stop()

try:
    data_path = get_data_path()
    profiles = build_customer_profiles(str(data_path))
except Exception as e:
    st.error("Müşteri profilleri oluşturulamadı.")
    st.code(str(e))
    st.stop()

client_ids = sorted(profiles["client_id"].dropna().unique().tolist())

st.sidebar.header("Ayarlar")
st.sidebar.write(f"Toplam müşteri sayısı: {len(client_ids)}")
st.sidebar.write(f"Veri dosyası: {data_path.name}")
st.sidebar.write(f"Model tipi: {model_type}")

selected_client = st.selectbox("Müşteri ID seçin", options=client_ids)
selected_profile = profiles.loc[profiles["client_id"] == selected_client].iloc[0].copy()

original_input = build_model_input(selected_profile, feature_order)
orig_pred, orig_score, orig_group, orig_advice = score_customer(model, model_type, original_input)

c1, c2, c3 = st.columns(3)
c1.metric("Orijinal Risk Skoru", f"{orig_score}/100")
c2.metric("Risk Grubu", orig_group)
c3.metric("Tahmin", "Yüksek Risk" if orig_pred == 1 else "Normal Risk")

st.markdown("## Müşteri Profili")
profile_view_cols = [
    "client_id",
    "amount_mean",
    "amount_std",
    "amount_max",
    "amount_sum",
    "yearly_income_first",
    "total_debt_first",
    "credit_score_first",
    "debt_to_income",
]
profile_view_cols = [c for c in profile_view_cols if c in selected_profile.index]
st.dataframe(pd.DataFrame(selected_profile[profile_view_cols]).T, use_container_width=True)

st.markdown("## What-if Simülasyonu")

with st.form("simulation_form"):
    s1, s2, s3 = st.columns(3)

    yearly_income_manual = s1.number_input(
        "Yıllık Gelir",
        min_value=0.0,
        value=float(selected_profile.get("yearly_income_first", 0)),
        step=1000.0
    )

    total_debt_manual = s2.number_input(
        "Toplam Borç",
        min_value=0.0,
        value=float(selected_profile.get("total_debt_first", 0)),
        step=1000.0
    )

    credit_score_manual = s3.number_input(
        "Kredi Skoru",
        min_value=0,
        max_value=1000,
        value=int(selected_profile.get("credit_score_first", 0)),
        step=1
    )

    amount_mean_manual = st.number_input(
        "Ortalama İşlem Tutarı",
        min_value=0.0,
        value=float(selected_profile.get("amount_mean", 0)),
        step=10.0
    )

    submitted = st.form_submit_button("Yeni Risk Hesapla")

if submitted:
    simulated_profile = selected_profile.copy()
    simulated_profile["yearly_income_first"] = yearly_income_manual if yearly_income_manual != 0 else 1
    simulated_profile["total_debt_first"] = total_debt_manual
    simulated_profile["credit_score_first"] = credit_score_manual
    simulated_profile["amount_mean"] = amount_mean_manual

    simulated_input = build_model_input(simulated_profile, feature_order)

    try:
        sim_pred, sim_score, sim_group, sim_advice = score_customer(model, model_type, simulated_input)

        r1, r2, r3 = st.columns(3)
        r1.metric("Yeni Risk Skoru", f"{sim_score}/100", delta=sim_score - orig_score)
        r2.metric("Yeni Risk Grubu", sim_group)
        r3.metric("Yeni Tahmin", "Yüksek Risk" if sim_pred == 1 else "Normal Risk")

        st.write(f"**Öneri:** {sim_advice}")

        compare_df = pd.DataFrame({
            "Alan": ["Risk Skoru", "Risk Grubu", "Tahmin"],
            "Orijinal": [orig_score, orig_group, "Yüksek Risk" if orig_pred == 1 else "Normal Risk"],
            "Yeni": [sim_score, sim_group, "Yüksek Risk" if sim_pred == 1 else "Normal Risk"],
        })
        st.dataframe(compare_df, use_container_width=True)

        with st.expander("Modele giden simülasyon verisi"):
            st.dataframe(simulated_input, use_container_width=True)

    except Exception as e:
        st.error("Simülasyon sırasında hata oluştu.")
        st.code(str(e))