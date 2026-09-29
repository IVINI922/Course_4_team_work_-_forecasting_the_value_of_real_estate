import streamlit as st
import plotly.express as px

from src.features import fmt_num


def render(df, comparison_df, meta, tuning_dir):
    st.header("Дашборд по данным")

    try:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Объектов", fmt_num(len(df)))
        c2.metric("Средняя цена", fmt_num(df['price'].mean()))
        c3.metric("Медиана цены", fmt_num(df['price'].median()))
        c4.metric("Уникальных городов",
                  fmt_num(df['city'].nunique()) if "city" in df.columns else "—")
    except Exception as e:
        st.warning(f"Метрики недоступны: {e}")

    try:
        p99 = df["price"].quantile(0.99)
        df_price = df[df["price"] <= p99]

        st.subheader("Распределение цены")
        st.caption(f"Показаны объекты с ценой ≤ {fmt_num(p99)} TRY "
                   f"(99-й перцентиль). Объектов: {fmt_num(len(df_price))} "
                   f"из {fmt_num(len(df))}.")
        st.plotly_chart(
            px.histogram(df_price, x="price", nbins=80,
                         title=f"Распределение цены (0 – {fmt_num(p99)} TRY)"),
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"График цены недоступен: {e}")

    try:
        st.subheader("Площадь vs Цена")

        c1, c2 = st.columns(2)
        with c1:
            max_size = st.slider(
                "Макс. площадь (м²)",
                min_value=50, max_value=500,        # ← реалистичный диапазон
                value=200, step=10,
)
        with c2:
            max_price = st.slider(
                "Макс. цена (TRY)",
                min_value=1_000_000, max_value=100_000_000,
                value=100_000_000, step=1_000_000,
                format="%d",
            )

        df_scatter = df[
            (df["price"] <= max_price) & (df["size"] <= max_size)
        ]
        st.caption(
            f"Объектов: {fmt_num(len(df_scatter))} "
            f"из {fmt_num(len(df))}."
        )
        fig = px.scatter(
            df_scatter, x="size", y="price",
            color="sub_type", opacity=0.5,
            title=f"Площадь vs Цена (≤ {fmt_num(max_size)} м², "
                  f"≤ {fmt_num(max_price)} TRY)",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"График «Площадь vs Цена» недоступен: {e}")

    try:
        if "city" in df.columns:
            st.subheader("Средняя цена по городам (топ-15)")
            agg = (df.groupby("city")["price"].mean()
                     .sort_values(ascending=False).head(15).reset_index())
            st.plotly_chart(px.bar(agg, x="city", y="price"),
                            use_container_width=True)
    except Exception as e:
        st.warning(f"График по городам недоступен: {e}")

    try:
        st.subheader("Средняя цена по типу жилья")
        agg2 = df.groupby("sub_type")["price"].mean().reset_index()
        st.plotly_chart(px.bar(agg2, x="sub_type", y="price"),
                        use_container_width=True)
    except Exception as e:
        st.warning(f"График по типу жилья недоступен: {e}")

    # ---------- СРАВНЕНИЕ МОДЕЛЕЙ ----------
    st.subheader("Сравнение моделей")
    if comparison_df.empty:
        st.warning(f"Файлы моделей не найдены в `{tuning_dir}`.")
    else:
        try:
            st.dataframe(comparison_df.style.format({"R2": "{:.4f}"}, na_rep="—"),
                         use_container_width=True)
        except Exception:
            st.dataframe(comparison_df, use_container_width=True)

        try:
            comp_plot = comparison_df.dropna(subset=["R2"]).copy()
            if not comp_plot.empty:
                fig = px.bar(comp_plot, x="Модель", y="R2",
                             title="R² по моделям (выше — лучше)",
                             color="R2", color_continuous_scale="Blues",
                             text="R2")
                fig.update_traces(texttemplate="%{text:.3f}",
                                  textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"График R² недоступен: {e}")

        with st.expander("Откуда взяты метрики"):
            st.markdown("""
            - **model_metadata.json** — метрики финальной модели (Gradient Boosting)
              на тестовой выборке.
            - **tuning_cache/*.pkl** — результаты подбора гиперпараметров.
              Если в `.pkl` сохранён словарь с `metrics` — показываются метрики
              на тесте. Если сохранён `RandomizedSearchCV` — показывается
              `best_score_` (CV R²).
            - **R²** — коэффициент детерминации (чем выше, тем лучше).
            """)

    st.subheader("Метрики финальной модели (подробно)")
    try:
        st.json(meta["metrics"])
    except Exception as e:
        st.warning(f"Метрики недоступны: {e}")
