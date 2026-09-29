import re
import numpy as np
import pandas as pd

from src.config import TOM_MAX, AGE_MAP


def fmt_num(x) -> str:
    """
    Форматирует число с пробелами между тысячами: 15751834 → '15 751 834'.
    Округляет до целого. При ошибке возвращает '—'.
    """
    try:
        n = int(round(float(x)))
    except (TypeError, ValueError):
        return "—"
    # форматирование с пробелами: 1234567 → '1 234 567'
    return f"{n:,}".replace(",", " ")


def te_lookup(encoder: dict, value) -> float:
    if not isinstance(encoder, dict):
        return 0.0
    v = str(value).strip() if value is not None else ""
    if v in encoder:
        return float(encoder[v])
    for k, val in encoder.items():
        if str(k).strip() == v:
            return float(val)
    vals = [val for val in encoder.values() if isinstance(val, (int, float))]
    return float(np.mean(vals)) if vals else 0.0


def parse_building_age(value) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    s = str(value).strip()
    if s in AGE_MAP:
        return float(AGE_MAP[s])
    if re.fullmatch(r"\d+", s):
        return float(s)
    return 0.0


def clamp_tom(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return float(min(max(v, 0.0), TOM_MAX))


def build_feature_row(raw: dict, scaler, target_encoders,
                     features: list, scale_cols: list) -> pd.DataFrame:
    row = {}
    row["size"] = float(raw["size"])
    row["building_age"] = parse_building_age(raw["building_age"])
    row["tom"] = clamp_tom(raw["tom"])

    row["city_te"] = te_lookup(target_encoders["city_target_encoder"], raw["city"])
    row["room_count_te"] = te_lookup(target_encoders["room_target_encoder"], raw["room_count"])
    row["district_grouped_te"] = te_lookup(target_encoders["district_target_encoder"], raw["district_grouped"])

    row["sub_type_New"]       = int(raw["sub_type"] == "New")
    row["sub_type_Secondary"] = int(raw["sub_type"] == "Secondary")
    row["heating_type_Central"]    = int(raw["heating_type"] == "Central")
    row["heating_type_Individual"] = int(raw["heating_type"] == "Individual")
    row["heating_type_None"]       = int(raw["heating_type"] == "None")

    row["price_currency_RUB"] = 1
    row["price_currency_USD"] = 0

    X = pd.DataFrame([row]).reindex(columns=features, fill_value=0)
    X_scaled = X.copy()
    X_scaled[scale_cols] = scaler.transform(X[scale_cols].to_numpy())
    return X_scaled