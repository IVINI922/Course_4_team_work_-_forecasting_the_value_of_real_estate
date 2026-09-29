import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import ART, TUNING_DIR, DATA_PATH


@st.cache_resource
def load_artifacts():
    """Загрузка модели и препроцессоров. При ошибке — понятное сообщение."""
    required = [
        "final_model_gb.joblib", "scaler.joblib", "onehot_encoder.joblib",
        "target_encoders.pkl", "feature_names.json", "model_metadata.json",
    ]
    missing = [f for f in required if not (ART / f).exists()]
    if missing:
        st.error(
            f"**Не найдены файлы артефактов** в папке `{ART}`:\n\n"
            + "\n".join(f"- `{m}`" for m in missing)
            + "\n\nУбедитесь, что папка `artifacts/` лежит рядом с `app.py`."
        )
        st.stop()

    try:
        model  = joblib.load(ART / "final_model_gb.joblib")
        scaler = joblib.load(ART / "scaler.joblib")
        ohe    = joblib.load(ART / "onehot_encoder.joblib")
        te     = joblib.load(ART / "target_encoders.pkl")
        feats  = json.load(open(ART / "feature_names.json", encoding="utf-8"))
        meta   = json.load(open(ART / "model_metadata.json", encoding="utf-8"))
    except Exception as e:
        st.error(f"**Ошибка загрузки артефактов:** `{type(e).__name__}: {e}`")
        st.stop()

    try:
        for enc_name, enc in te.items():
            if isinstance(enc, dict):
                te[enc_name] = {str(k).strip(): float(v) for k, v in enc.items()}
    except Exception as e:
        st.warning(f"Не удалось нормализовать target-энкодеры: {e}")

    return model, scaler, ohe, te, feats, meta


@st.cache_data
def load_data():
    if not Path(DATA_PATH).exists():
        st.error(f"**Файл данных не найден:** `{DATA_PATH}`")
        st.stop()

    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
    except Exception as e:
        st.error(f"**Ошибка чтения `{DATA_PATH}`:** `{type(e).__name__}: {e}`")
        st.stop()

    if "address" not in df.columns:
        st.error("**В датасете нет колонки `address`.**")
        st.stop()

    try:
        addr = df["address"].astype("string")
        parts = addr.str.split("/", n=2, expand=True)
        df["city"] = parts[0].str.strip()
        df["district_grouped"] = parts[1].str.strip() if parts.shape[1] > 1 else None
    except Exception as e:
        st.warning(f"Не удалось разобрать `address`: {e}")

    return df


@st.cache_data
def load_model_comparison(_meta):
    """Читает tuning_cache/*.pkl. _meta — метаданные финальной модели."""
    rows = []
    if not TUNING_DIR.exists():
        return pd.DataFrame()

    for pkl_file in sorted(TUNING_DIR.glob("*.pkl")):
        name = pkl_file.stem.replace("_", " ")
        row = {
            "Модель": name,
            "R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan,
            "Источник": "нет данных",
        }
        obj = None
        try:
            with open(pkl_file, "rb") as f:
                obj = pickle.load(f)
        except Exception:
            try:
                obj = joblib.load(pkl_file)
            except Exception:
                rows.append(row)
                continue

        if obj is None:
            rows.append(row)
            continue

        if isinstance(obj, dict):
            metrics = obj.get("metrics") or obj.get("test_metrics") or {}
            if isinstance(metrics, dict) and metrics:
                row["R2"]   = metrics.get("R2")   or metrics.get("R2_log")
                row["MAE"]  = metrics.get("MAE")  or metrics.get("MAE_log")
                row["RMSE"] = metrics.get("RMSE") or metrics.get("RMSE_log")
                row["MAPE"] = metrics.get("MAPE") or metrics.get("MAPE_pct")
                row["Источник"] = "metrics"
            elif "best_score" in obj:
                row["R2"] = obj["best_score"]
                row["Источник"] = "best_score (CV R²)"
            rows.append(row)
            continue

        if hasattr(obj, "best_score_"):
            row["R2"] = float(obj.best_score_)
            row["Источник"] = "best_score_ (CV R²)"
        elif hasattr(obj, "score"):
            row["Источник"] = "модель (R² нет на тесте)"
        rows.append(row)

    if _meta.get("metrics"):
        m = _meta["metrics"]
        rows.append({
            "Модель": _meta.get("model_name", "Gradient Boosting"),
            "R2":   m.get("R2_log"), "MAE":  m.get("MAE_log"),
            "RMSE": m.get("RMSE_log"), "MAPE": m.get("MAPE_pct"),
            "Источник": "model_metadata.json (test)",
        })

    comp = pd.DataFrame(rows)
    if "R2" in comp.columns:
        comp = comp.sort_values("R2", ascending=False, na_position="last").reset_index(drop=True)
    return comp