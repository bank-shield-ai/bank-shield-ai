import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "notebooks" / "df_combined.parquet"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_OUTPUT = MODEL_DIR / "behavioral_risk_regressor.joblib"


def clean_to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce"
    ).fillna(0)


def build_customer_profiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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

    df["is_night_transaction"] = df["hour"].apply(
        lambda x: 1 if pd.notnull(x) and 0 <= x <= 6 else 0
    ) if "hour" in df.columns else 0

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


def minmax_0_100(series: pd.Series) -> pd.Series:
    s = series.copy()
    s_min = s.min()
    s_max = s.max()
    if s_max == s_min:
        return pd.Series(np.full(len(s), 50.0), index=s.index)
    return ((s - s_min) / (s_max - s_min)) * 100


def build_continuous_target(customer_df: pd.DataFrame) -> pd.DataFrame:
    df = customer_df.copy()

    # 1) debt_to_income etkisi: log ile yumuşat
    dti_component = np.log1p(df["debt_to_income"].clip(lower=0))
    dti_score = minmax_0_100(dti_component)

    # 2) credit score ters etki: düşük skor = yüksek risk
    credit_clipped = df["credit_score_first"].clip(lower=300, upper=850)
    credit_inverse = 850 - credit_clipped
    credit_score_component = minmax_0_100(credit_inverse)

    # 3) fraud rate
    fraud_component = minmax_0_100(df["is_fraud_mean"].clip(lower=0))

    # 4) night / fast davranışları
    night_component = minmax_0_100(df["is_night_transaction_mean"].clip(lower=0))
    fast_component = minmax_0_100(df["fast_tx_mean"].clip(lower=0))

    # 5) amount_mean hafif katkı
    amount_component = minmax_0_100(np.log1p(df["amount_mean"].clip(lower=0)))

    # Ağırlıklı, sürekli risk skoru
    df["risk_score_target"] = (
        0.45 * dti_score +
        0.25 * credit_score_component +
        0.12 * fraud_component +
        0.08 * night_component +
        0.05 * fast_component +
        0.05 * amount_component
    )

    df["risk_score_target"] = df["risk_score_target"].clip(0, 100)

    return df


def get_risk_group(score: float) -> str:
    if score < 30:
        return "DÜŞÜK"
    elif score < 70:
        return "ORTA"
    return "YÜKSEK"


def main():
    print("Veri okunuyor...")
    df = pd.read_parquet(DATA_PATH)

    print("Müşteri profilleri oluşturuluyor...")
    customer_df = build_customer_profiles(df)

    print("Sürekli hedef skor oluşturuluyor...")
    customer_df = build_continuous_target(customer_df)

    feature_cols = [
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

    X = customer_df[feature_cols].copy()
    y = customer_df["risk_score_target"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("Model eğitiliyor...")
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test).clip(0, 100)

    print("\nModel performansı")
    print("-" * 30)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"R2 : {r2_score(y_test, y_pred):.3f}")

    preview = pd.DataFrame({
        "Gerçek Skor": y_test.values[:10],
        "Tahmin Skor": y_pred[:10],
        "Tahmin Grup": [get_risk_group(v) for v in y_pred[:10]]
    })
    print("\nİlk 10 tahmin:")
    print(preview)

    payload = {
        "model": model,
        "features": feature_cols,
        "model_type": "regressor",
        "target_name": "risk_score_target",
        "notes": "Continuous 0-100 behavioral risk score model"
    }

    joblib.dump(payload, MODEL_OUTPUT)
    print(f"\nModel kaydedildi: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()