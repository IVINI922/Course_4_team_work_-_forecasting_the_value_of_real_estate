"""Точка входа Streamlit-приложения «Прогноз стоимости недвижимости»."""
import streamlit as st

from src.config import TUNING_DIR
from src.data import load_artifacts, load_data, load_model_comparison
from src.ui import predict, dashboard, help_tab

st.set_page_config(
    page_title="Прогноз стоимости недвижимости",
    page_icon="🏠",
    layout="wide",
)

# --- артефакты и данные (кэшируются) ---
model, scaler, ohe, target_encoders, feats_meta, META = load_artifacts()
FEATURES = feats_meta["final_features"]
SCALE_COLS = list(getattr(scaler, "feature_names_in_",
                          ["size", "building_age", "tom"]))

df = load_data()
comparison_df = load_model_comparison(META)

# --- вкладки ---
tab_predict, tab_dash, tab_help = st.tabs(["🏠 Прогноз", "📊 Дашборд", "ℹ️ Справка"])

with tab_predict:
    predict.render(df, model, scaler, target_encoders,
                   FEATURES, SCALE_COLS, META)

with tab_dash:
    dashboard.render(df, comparison_df, META, TUNING_DIR)

with tab_help:
    help_tab.render(META)