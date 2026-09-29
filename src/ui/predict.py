import numpy as np
import streamlit as st

from src.config import TOM_MAX, AGE_OPTIONS
from src.features import (
    fmt_num, parse_building_age, clamp_tom, te_lookup, build_feature_row,
)

# Курс конвертации TRY → RUB
TRY_TO_RUB = 1.73


def render(df, model, scaler, target_encoders, features, scale_cols, meta):
    st.header("Прогноз стоимости объекта")
    st.caption(
        "Форма содержит все характеристики объекта. Признаки, помеченные "
        "⚠, в текущей версии модели не влияют на прогноз (обоснование — "
        "в разделе «Справка»). Прогноз выводится в рублях (конвертация "
        f"из TRY по курсу 1 TRY = {TRY_TO_RUB} RUB)."
    )

    cities = sorted(df["city"].dropna().unique().tolist()) if "city" in df.columns else []
    districts = sorted(df["district_grouped"].dropna().unique().tolist()) if "district_grouped" in df.columns else []
    rooms = sorted(target_encoders["room_target_encoder"].keys())

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            size = st.number_input("Площадь, м²", 5.0, 2000.0, 100.0, 1.0)
            room_count = st.selectbox("Количество комнат ⚠", rooms,
                                      help="Feature importance 2,6%, ноль сплитов.")
            building_age = st.selectbox("Возраст здания", AGE_OPTIONS)

        with c2:
            city = st.selectbox("Город ⚠", cities if cities else ["—"],
                                help="Пороги сплитов выше максимума city_te.")
            district_grouped = st.selectbox("Район ⚠", districts if districts else ["—"],
                                             help="Feature importance 0,3%.")
            tom = st.number_input(
                f"TOM (время экспозиции, дней, ≤ {TOM_MAX})",
                0, TOM_MAX, min(int(df["tom"].median()), TOM_MAX),
                help=f"Значения выше {TOM_MAX} не меняют прогноз.",
            )

        with c3:
            sub_type = st.selectbox("Тип жилья", ["New", "Secondary"])
            heating_type = st.selectbox("Отопление", ["Central", "Individual", "None"])

        submitted = st.form_submit_button("Рассчитать", use_container_width=True)

    if not submitted:
        return

    errors = []
    if size <= 0:
        errors.append("Площадь должна быть > 0.")
    if size > 2000:
        errors.append("Площадь не может превышать 2000 м².")
    if errors:
        for e in errors:
            st.error(e)
        return

    raw = {
        "size": size, "building_age": building_age, "tom": tom,
        "sub_type": sub_type, "heating_type": heating_type,
        "city": city, "room_count": room_count,
        "district_grouped": district_grouped,
    }

    try:
        X = build_feature_row(raw, scaler, target_encoders, features, scale_cols)
    except Exception as e:
        st.error(f"**Ошибка подготовки фич:** `{type(e).__name__}: {e}`")
        return

    try:
        log_pred = float(model.predict(X)[0])
        pred_try = float(np.expm1(log_pred))
    except Exception as e:
        st.error(f"**Ошибка модели:** `{type(e).__name__}: {e}`")
        return

    # --- конвертация TRY → RUB ---
    pred_rub = pred_try * TRY_TO_RUB

    mape = meta["metrics"]["MAPE_pct"]
    delta_rub = pred_rub * mape / 100.0

    st.success(f"### Прогнозируемая цена: {fmt_num(pred_rub)} ₽")
    st.info(
        f"Диапазон ошибки (±{mape:.1f}%): "
        f"{fmt_num(pred_rub - delta_rub)} — {fmt_num(pred_rub + delta_rub)} ₽"
    )
    st.caption(
        f"Модель предсказывает цену в турецких лирах: **{fmt_num(pred_try)} TRY**. "
        f"Пересчёт в рубли по курсу **1 TRY = {TRY_TO_RUB} RUB**."
    )
    st.caption(
        "⚠ Город, район и количество комнат **не влияют** на прогноз "
        "в текущей версии модели — это ограничение артефактов."
    )

    with st.expander("Входные данные и преобразованные фичи"):
        st.write("Сырые данные:", raw)
        st.write({
            "building_age (в модель)": parse_building_age(building_age),
            "tom (в модель, после clamp)": clamp_tom(tom),
            "city_te": round(te_lookup(target_encoders["city_target_encoder"], city), 4),
            "room_count_te": round(te_lookup(target_encoders["room_target_encoder"], room_count), 4),
            "district_grouped_te": round(te_lookup(target_encoders["district_target_encoder"], district_grouped), 4),
            "price_currency_RUB (фиксировано)": 1,
        })
        st.dataframe(X.T.rename(columns={0: "scaled_value"}))

        st.markdown("**Промежуточные значения:**")
        st.write({
            "pred (log)": round(log_pred, 6),
            "pred (TRY)": round(pred_try, 2),
            f"pred (RUB, ×{TRY_TO_RUB})": round(pred_rub, 2),
        })