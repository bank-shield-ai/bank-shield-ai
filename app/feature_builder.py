import pandas as pd
import numpy as np
import ast

DAY_NAME_MAP = {
    "Pazartesi": "Monday",
    "Salı": "Tuesday",
    "Çarşamba": "Wednesday",
    "Perşembe": "Thursday",
    "Cuma": "Friday",
    "Cumartesi": "Saturday",
    "Pazar": "Sunday",
}

DAY_NUM_MAP = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


def normalize_text(value):
    if pd.isna(value):
        return "None"
    return str(value).strip()


def safe_scalar(value, default=np.nan):
    if isinstance(value, pd.Series):
        if len(value) == 0:
            return default
        return value.iloc[0]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return value.flatten()[0]
    if isinstance(value, list):
        if len(value) == 0:
            return default
        return value[0]
    return value


def safe_float(value, default=0.0):
    value = safe_scalar(value, default=default)
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_text(value, default="Unknown"):
    value = safe_scalar(value, default=default)
    if pd.isna(value):
        return default
    return str(value).strip()


def robust_median(series, fallback):
    s = pd.to_numeric(series, errors="coerce")
    med = s.dropna().median()
    if pd.isna(med):
        return fallback
    return float(med)


def parse_last3(value):
    value = safe_scalar(value, default=[])
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return []


def derive_time_flags(hour: int, day_name_tr: str):
    is_weekend = int(day_name_tr in ["Cumartesi", "Pazar"])
    is_night = int(hour < 6 or hour >= 23)
    is_peak_hour = int(hour in [12, 13, 18, 19, 20])
    return is_weekend, is_night, is_peak_hour


def find_client_row(client_id, client_summary_df):
    row = client_summary_df.loc[
        client_summary_df["client_id"].astype(str).str.strip() == str(client_id).strip()
    ]
    if row.empty:
        return None
    return row.iloc[0]


def find_merchant_risk_score(mcc, merchant_risk_df):
    mcc = normalize_text(mcc)
    row = merchant_risk_df.loc[
        merchant_risk_df["mcc"].astype(str).str.strip() == mcc
    ]
    if row.empty:
        return 0.02
    if "merchant_risk_score" in merchant_risk_df.columns:
        return safe_float(row["merchant_risk_score"].iloc[0], default=0.02)
    if "fraud_rate" in merchant_risk_df.columns:
        return safe_float(row["fraud_rate"].iloc[0], default=0.02)
    if "merchant_risk_flag" in merchant_risk_df.columns:
        return 0.10 if safe_float(row["merchant_risk_flag"].iloc[0], default=0.0) >= 1 else 0.02
    return 0.02


def build_features(
    client_id,
    amount,
    islem_tipi,
    hour,
    day,
    month,
    year,
    haftanin_gunu_tr,
    merchant_city,
    merchant_state,
    mcc,
    is_return,
    tx_datetime,
    client_summary_df,
    merchant_risk_df
):
    global_user_mean_amount = robust_median(client_summary_df["user_mean_amount"], 50.0)
    global_user_std_amount = robust_median(client_summary_df["user_std_amount"], 25.0)
    global_current_age = robust_median(client_summary_df["current_age"], 40.0)
    global_yearly_income = robust_median(client_summary_df["yearly_income"], 50000.0)
    global_total_debt = robust_median(client_summary_df["total_debt"], 5000.0)
    global_credit_score = robust_median(client_summary_df["credit_score"], 650.0)
    global_per_capita_income = robust_median(client_summary_df["per_capita_income"], 25000.0)
    global_credit_limit = robust_median(client_summary_df["credit_limit"], 3000.0)

    client_row = find_client_row(client_id, client_summary_df)

    if client_row is None:
        user_tx_count = 0
        user_mean_amount = global_user_mean_amount
        user_std_amount = global_user_std_amount
        last_tx_time = None
        last_3_amounts = []
        credit_limit = global_credit_limit
        current_age = global_current_age
        yearly_income = global_yearly_income
        total_debt = global_total_debt
        credit_score = global_credit_score
        gender = "Unknown"
        per_capita_income = global_per_capita_income
    else:
        user_tx_count = int(safe_float(client_row.get("user_tx_count", 0), default=0))
        user_mean_amount = safe_float(client_row.get("user_mean_amount", global_user_mean_amount), default=global_user_mean_amount)
        user_std_amount = safe_float(client_row.get("user_std_amount", global_user_std_amount), default=global_user_std_amount)
        if user_std_amount <= 0:
            user_std_amount = max(global_user_std_amount, 1.0)

        last_tx_time = pd.to_datetime(safe_scalar(client_row.get("last_tx_time", pd.NaT), default=pd.NaT), errors="coerce")
        last_3_amounts = parse_last3(client_row.get("last_3_amounts", []))

        credit_limit = safe_float(client_row.get("credit_limit", global_credit_limit), default=global_credit_limit) or global_credit_limit
        current_age = safe_float(client_row.get("current_age", global_current_age), default=global_current_age) or global_current_age
        yearly_income = safe_float(client_row.get("yearly_income", global_yearly_income), default=global_yearly_income) or global_yearly_income
        total_debt = safe_float(client_row.get("total_debt", global_total_debt), default=global_total_debt)
        credit_score = safe_float(client_row.get("credit_score", global_credit_score), default=global_credit_score) or global_credit_score
        gender = safe_text(client_row.get("gender", "Unknown"), default="Unknown")
        per_capita_income = safe_float(client_row.get("per_capita_income", global_per_capita_income), default=global_per_capita_income) or global_per_capita_income

    amount = safe_float(amount, default=0.0)
    hour = int(safe_float(hour, default=0))
    day = int(safe_float(day, default=1))
    month = int(safe_float(month, default=1))
    year = int(safe_float(year, default=2010))

    merchant_city = normalize_text(merchant_city)
    merchant_state = normalize_text(merchant_state)
    mcc = normalize_text(mcc)
    use_chip = normalize_text(islem_tipi)

    log_amount = np.log1p(amount)
    high_amount = int(amount > 100)
    amount_to_limit_ratio = amount / (credit_limit + 1e-6)
    amount_zscore = (amount - user_mean_amount) / (user_std_amount + 1e-6)
    amount_deviation = amount - user_mean_amount

    if last_tx_time is not None and pd.notna(last_tx_time):
        time_diff = (tx_datetime - last_tx_time).total_seconds()
        if time_diff <= 0:
            time_diff = 999999.0
    else:
        time_diff = 999999.0

    fast_tx = int(0 < time_diff < 300)
    very_fast_tx = int(0 < time_diff < 60)

    rolling_mean_3 = float(np.mean(last_3_amounts)) if len(last_3_amounts) > 0 else user_mean_amount
    rolling_std_3 = float(np.std(last_3_amounts, ddof=0)) if len(last_3_amounts) > 1 else 0.0
    rolling_amount_deviation = amount - rolling_mean_3

    day_of_week = DAY_NAME_MAP[haftanin_gunu_tr]
    day_of_week_num = DAY_NUM_MAP[day_of_week]
    is_weekend, is_night, is_peak_hour = derive_time_flags(hour, haftanin_gunu_tr)

    merchant_risk_score = min(max(find_merchant_risk_score(mcc, merchant_risk_df), 0.0), 0.30)
    merchant_risk_score_log = np.log1p(merchant_risk_score)

    eps = 1e-6
    amount_to_user_mean_ratio = amount / (abs(user_mean_amount) + eps)
    amount_to_rolling_mean_ratio = amount / (abs(rolling_mean_3) + eps)
    abs_amount_deviation = abs(amount_deviation)
    abs_rolling_amount_deviation = abs(rolling_amount_deviation)
    tx_velocity = 1 / (time_diff + 1)

    amount_spike_2std = int(amount > (user_mean_amount + 2 * user_std_amount))
    amount_spike_3std = int(amount > (user_mean_amount + 3 * user_std_amount))
    log_amount_to_income = np.log1p(amount) / (np.log1p(max(yearly_income, 0.0)) + eps)
    debt_to_income_ratio = total_debt / (yearly_income + eps)

    return pd.DataFrame([{
        "amount": amount,
        "log_amount": log_amount,
        "amount_zscore": amount_zscore,
        "high_amount": high_amount,
        "amount_to_limit_ratio": amount_to_limit_ratio,
        "hour": hour,
        "day": day,
        "month": month,
        "year": year,
        "day_of_week": day_of_week,
        "day_of_week_num": day_of_week_num,
        "is_weekend": is_weekend,
        "is_night": is_night,
        "is_peak_hour": is_peak_hour,
        "mcc": mcc,
        "merchant_city": merchant_city,
        "merchant_state": merchant_state,
        "is_return": int(is_return),
        "current_age": current_age,
        "yearly_income": yearly_income,
        "per_capita_income": per_capita_income,
        "total_debt": total_debt,
        "credit_score": credit_score,
        "gender": gender,
        "user_tx_count": user_tx_count,
        "user_mean_amount": user_mean_amount,
        "user_std_amount": user_std_amount,
        "amount_deviation": amount_deviation,
        "time_diff": time_diff,
        "fast_tx": fast_tx,
        "very_fast_tx": very_fast_tx,
        "rolling_mean_3": rolling_mean_3,
        "rolling_std_3": rolling_std_3,
        "rolling_amount_deviation": rolling_amount_deviation,
        "amount_to_user_mean_ratio": amount_to_user_mean_ratio,
        "amount_to_rolling_mean_ratio": amount_to_rolling_mean_ratio,
        "abs_amount_deviation": abs_amount_deviation,
        "abs_rolling_amount_deviation": abs_rolling_amount_deviation,
        "tx_velocity": tx_velocity,
        "merchant_risk_score": merchant_risk_score,
        "merchant_risk_score_log": merchant_risk_score_log,
        "amount_spike_2std": amount_spike_2std,
        "amount_spike_3std": amount_spike_3std,
        "log_amount_to_income": log_amount_to_income,
        "debt_to_income_ratio": debt_to_income_ratio,
        "use_chip": use_chip,
    }])