import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

plt.style.use("ggplot")

st.title("Визуализация данных")

st.markdown("В этом разделе будет представлена визуализация данных.")

st.subheader("Зависимость числовых признаков от целевого")

@st.cache_data
def load(path):
    return pd.read_csv(path)

df = load('./data/dum_data.csv').astype(np.float64).drop(['Unnamed: 0'], axis=1)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))
for idx, feature in enumerate(df.columns[:3]):
    df.plot(feature, "price_usd", subplots=True, kind="scatter", ax=axes[idx%3])
    
st.pyplot(fig)

fig = plt.figure()

sns.heatmap(df.loc[:,['odometer_value', 'engine_capacity', 'year_produced', 'price_usd']].corr(), annot=True)

st.pyplot(fig)

st.subheader("Статистика значений числовых и категориальных признаков")

yp = df["year_produced"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(3, 2))

ax.bar(yp.index, yp.values, width=1.2, edgecolor='black')

st.pyplot(fig, use_container_width=False)

st.caption("Распределение значений колонки `year_produced`")

st.divider()

df = load('./data/cars.csv')

fig, ax = plt.subplots(figsize=(3, 3))

ax.pie(df['engine_has_gas'].value_counts().values, labels=df['engine_has_gas'].value_counts().index)

st.pyplot(fig, use_container_width=False)

st.caption("Распределение значений колонки `engine_has_gas`")

st.divider()

fig = plt.figure()

sns.boxplot(data=df, x="odometer_value")

st.pyplot(fig)

st.caption("Диаграмма \"Ящик с усами\" для колонки `odometer_value`")