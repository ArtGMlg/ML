import streamlit as st
import pandas as pd
import numpy as np

st.title("Всё о данных")

st.text("Представление первоначальных данных:")

st.dataframe(pd.read_csv("./data/cars.csv"))

st.subheader("Описание")

st.markdown("Набор данных собран с различных веб-ресурсов с целью изучения рынка подержанных автомобилей и попытки построить модель, которая эффективно прогнозирует цену автомобиля на основе его параметров (как числовых, так и категориальных).")

st.markdown("Однако перед началом работы с датасетом его необходимо предобработать (заполнитть пропуски, преобразовать категориальные признаки). Кроме этого было принято решение отбросить столбцы, которые не оказывали влияния на результат работы эстиматоров")

st.subheader("Предобработка")

st.markdown("В наборе данных имели место быть пропуски в столбце `engine_capacity`. Поскольку он содержит действительные числа, было решено заполнить пропуски _средним по столбцу_.")

st.markdown("Над столбцами `transmission`, `engine_type`, `has_warranty`, `is_exchangeable`, `engine_has_gas`, `body_type`, `state`, `engine_fuel`, `drivetrain`, `location_region`, `feature_0`, `feature_1`, `feature_2`, `feature_3`, `feature_4`, `feature_5`, `feature_6`, `feature_7`, `feature_8`, `feature_9` было произведено one-hot кодирование.")

st.text("Результат предобработки:")

st.dataframe(pd.read_csv("./data/dum_data.csv"))
