import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import base64

st.set_page_config(
    page_title="CaixaBank AI Risk Hub - Risk Skoru",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# YARDIMCILAR
# -----------------------------
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None


def risk_group(score: float) -> str:
    if score < 30:
        return "DÜŞÜK RİSK"
    elif score < 70:
        return "ORTA RİSK"
    return "YÜKSEK RİSK"


def recommendation(score: float) -> str:
    if score < 30:
        return "Standart izleme yeterli."
    elif score < 70:
        return "Ek kontrol ve periyodik takip önerilir."
    return "Manuel inceleme ve sıkı doğrulama önerilir."


# -----------------------------
# PATH
# -----------------------------
CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parent.parent
PROJECT_DIR = APP_DIR.parent

MODELS_DIR = PROJECT_DIR / "models"
SUMMARY_DATA_PATH = PROJECT_DIR / "data" / "processed" / "customer_risk_summary.parquet"
logo_path = APP_DIR / "logo.png"

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

view_mode = st.session_state.get("behavioral_view_mode", "customer")
is_authorized = view_mode == "authorized"

logo_base64 = get_base64_image(str(logo_path))
if logo_base64:
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="200">'
else:
    logo_html = '<h2 style="color: #004587; margin:0;">CaixaBank</h2>'

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 0.8rem;
    max-width: 1400px;
}

.header-box {
    background-color: white;
    padding: 12px 36px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 3px solid #ff6600;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    margin-bottom: 18px;
}

.hero-section {
    background: linear-gradient(135deg, #004587 0%, #002e5a 100%);
    color: white;
    padding: 42px 40px;
    border-radius: 0 0 36px 36px;
    margin-bottom: 28px;
    box-shadow: 0 10px 28px rgba(0, 69, 135, 0.18);
}

.section-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e7ebf0;
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    margin-bottom: 18px;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background-color: #004587;
    color: white;
    border-radius: 999px;
    border: none;
    font-weight: 700;
}

.footer-box {
    text-align: center;
    margin-top: 70px;
    padding: 28px;
    color: #7f8c8d;
    font-size: 0.85rem;
    border-top: 1px solid #e1e4e8;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
header_title = "Yetkili Modülü • Risk Skoru" if is_authorized else "Müşteri Modülü • Risk Skoru"

st.markdown(f"""
<div class="header-box">
    <div>{logo_html}</div>
    <div style="color:#004587;font-weight:700;">{header_title}</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero-section">
    <h1 style="color:white; margin-bottom: 8px;">Risk Skoru</h1>
    <p style="opacity:0.92; margin-bottom:0;">
        {"Yetkili kullanıcılar için gelişmiş risk analizi ve simülasyon ekranı." if is_authorized else "Müşteri kullanıcılar için senaryo bazlı risk simülasyonu ekranı."}
    </p>
</div>
""", unsafe_allow_html=True)

top_left, top_right = st.columns([1, 3])

with top_left:
    if is_authorized:
        if st.button("← Yetkili Paneline Dön"):
            st.switch_page("pages/00_Yetkili_Paneli.py")
    else:
        if st.button("← Anasayfaya Dön"):
            st.switch_page("streamlit_app.py")

with top_right:
    st.markdown(
        '<div class="section-card"><b>Mod:</b> ' + ('Yetkili Girişi' if is_authorized else 'Müşteri Girişi') + '</div>',
        unsafe_allow_html=True
    )

# -----------------------------
# MODEL YÜKLEME
# -----------------------------
@st.cache_resource
def load_model_payload():
    checked_paths = []

    for filename in MODEL_CANDIDATES:
        model_path = MODELS_DIR / filename
        checked_paths.append(str(model_path))

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
                "model_path": str(model_path),
            }

    raise FileNotFoundError(
        f"Model bulunamadı. Kontrol edilen yollar: {checked_paths}"
    )


@st.cache_data
def load_customer_profiles() -> pd.DataFrame:
    if not SUMMARY_DATA_PATH.exists():
        raise FileNotFoundError(f"Özet veri dosyası bulunamadı: {SUMMARY_DATA_PATH}")

    df = pd.read_parquet(SUMMARY_DATA_PATH)

    if "client_id" not in df.columns:
        raise ValueError("Summary parquet içinde 'client_id' kolonu yok.")

    if "amount_sum" not in df.columns:
        df["amount_sum"] = df["amount_mean"] * 10

    required_base_cols = [
        "amount_mean",
        "amount_std",
        "amount_max",
        "is_night_transaction_mean",
        "fast_tx_mean",
        "yearly_income_first",
        "total_debt_first",
        "credit_score_first",
    ]

    for col in required_base_cols:
        if col not in df.columns:
            df[col] = 0

    if "debt_to_income" not in df.columns:
        income = df["yearly_income_first"].replace(0, 1)
        df["debt_to_income"] = df["total_debt_first"] / income

    df["amount_std"] = df["amount_std"].fillna(0)
    df["yearly_income_first"] = df["yearly_income_first"].replace(0, 1)

    return df


def build_model_input(profile_row: pd.Series, feature_order: list) -> pd.DataFrame:
    row = {}
    for feat in feature_order:
        row[feat] = profile_row.get(feat, 0)
    return pd.DataFrame([row])


def score_customer(model, model_type: str, X_input: pd.DataFrame):
    if model_type == "regressor":
        raw_score = float(model.predict(X_input)[0])
        score = max(0, min(100, round(raw_score, 1)))
        pred = 1 if score >= 60 else 0
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


try:
    payload = load_model_payload()
    model = payload["model"]
    feature_order = payload["features"]
    model_type = payload["model_type"]
    model_name = Path(payload["model_path"]).name
except Exception as e:
    st.error("Model yüklenemedi.")
    st.code(str(e))
    st.stop()

try:
    profiles = load_customer_profiles()
except Exception as e:
    st.error("Müşteri profilleri yüklenemedi.")
    st.code(str(e))
    st.stop()

client_ids = sorted(profiles["client_id"].dropna().unique().tolist())

with st.sidebar:
    st.header("Sistem Bilgisi")
    st.write(f"Toplam müşteri sayısı: {len(client_ids)}")
    st.write(f"Veri dosyası: {SUMMARY_DATA_PATH.name}")
    st.write(f"Model tipi: {model_type}")
    st.write(f"Model dosyası: {model_name}")

# -----------------------------
# MÜŞTERİ / YETKİLİ SEÇİM FARKI
# -----------------------------
if is_authorized:
    selected_client = st.selectbox("Müşteri ID seçin", options=client_ids)
else:
    selected_client = client_ids[0]
    st.caption("Simülasyon müşteri profili otomatik olarak yüklenmiştir.")

selected_profile = profiles.loc[profiles["client_id"] == selected_client].iloc[0].copy()

original_input = build_model_input(selected_profile, feature_order)
orig_pred, orig_score, orig_group, orig_advice = score_customer(
    model, model_type, original_input
)

# -----------------------------
# YETKİLİ MODUNDA PROFİL GÖRÜNSÜN
# -----------------------------
if is_authorized:
    c1, c2, c3 = st.columns(3)
    c1.metric("Orijinal Risk Skoru", f"{orig_score}/100")
    c2.metric("Risk Grubu", orig_group)
    c3.metric("Tahmin", "Yüksek Risk" if orig_pred == 1 else "Normal Risk")

    st.markdown("## Müşteri Profili")
    profile_view_cols = [
        "client_id",
        "yearly_income_first",
        "total_debt_first",
        "credit_score_first",
        "amount_mean",
        "amount_std",
        "amount_max",
        "amount_sum",
        "is_night_transaction_mean",
        "fast_tx_mean",
        "debt_to_income",
    ]
    profile_view_cols = [c for c in profile_view_cols if c in selected_profile.index]

    st.dataframe(
        pd.DataFrame(selected_profile[profile_view_cols]).T,
        use_container_width=True
    )

# -----------------------------
# WHAT-IF SİMÜLASYONU
# -----------------------------
st.markdown("## Ya Şöyle Olsa? Simülasyonu")
st.write("Aşağıdaki alanları değiştirerek yeni risk skorunu hesaplayabilirsiniz.")

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

    simulated_profile["debt_to_income"] = (
        simulated_profile["total_debt_first"] / simulated_profile["yearly_income_first"]
    )

    if "amount_sum" not in simulated_profile.index or pd.isna(simulated_profile["amount_sum"]):
        simulated_profile["amount_sum"] = simulated_profile["amount_mean"] * 10

    simulated_input = build_model_input(simulated_profile, feature_order)

    try:
        sim_pred, sim_score, sim_group, sim_advice = score_customer(
            model, model_type, simulated_input
        )

        r1, r2, r3 = st.columns(3)
        score_delta = round(sim_score - orig_score, 2)

        r1.metric(
            "Yeni Risk Skoru",
            f"{sim_score:.1f}/100",
            delta=score_delta,
            delta_color="inverse"
        )
        r2.metric("Yeni Risk Grubu", sim_group)
        r3.metric("Yeni Tahmin", "Yüksek Risk" if sim_pred == 1 else "Normal Risk")

        if sim_score < 30:
            st.success(f"Yeni Risk Grubu: {sim_group}")
        elif sim_score < 70:
            st.warning(f"Yeni Risk Grubu: {sim_group}")
        else:
            st.error(f"Yeni Risk Grubu: {sim_group}")

        st.write(f"**Öneri:** {sim_advice}")

        compare_df = pd.DataFrame({
            "Alan": ["Risk Skoru", "Risk Grubu", "Tahmin", "Borç / Gelir Oranı"],
            "Orijinal": [
                orig_score,
                orig_group,
                "Yüksek Risk" if orig_pred == 1 else "Normal Risk",
                round(float(selected_profile.get("debt_to_income", 0)), 4),
            ],
            "Yeni": [
                sim_score,
                sim_group,
                "Yüksek Risk" if sim_pred == 1 else "Normal Risk",
                round(float(simulated_profile.get("debt_to_income", 0)), 4),
            ],
        })
        st.dataframe(compare_df, use_container_width=True)

        with st.expander("Modele giden simülasyon verisi"):
            st.dataframe(simulated_input, use_container_width=True)

    except Exception as e:
        st.error("Simülasyon sırasında hata oluştu.")
        st.code(str(e))

st.markdown("""
<div class="footer-box">
    © 2026 CaixaBank AI Risk Management Solutions. All Rights Reserved.
</div>
""", unsafe_allow_html=True)