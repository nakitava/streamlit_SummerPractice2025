import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix

st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffebee;  /* Светло-розовый цвет */
    }
    /* Темно-красный цвет для всего текста */
    h1, h2, h3, h4, h5, h6, p, div, span, .stMarkdown, .stMetric, .stAlert {
        color: #8B0000 !important;
    }
    /* Стили для метрик */
    .stMetric {
        border: 1px solid #8B0000;
        border-radius: 10px;
        padding: 10px;
    }
    /* Стили для заголовков графиков */
    .stPlotlyChart h2 {
        color: #8B0000 !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Настройки страницы
st.set_page_config(layout="wide", page_title="Дашборд классификации студентов")
st.title("StudentsDropout \n2023-ФГиИБ-ПИ-1б - Золотарева С.А.")

# 1. Описание набора данных
st.header("Описание набора данных")
st.markdown("""
<span style='color:#8B0000'>
Набор данных содержит информацию о студентах, 
собранную для анализа успеваемости и факторов, влияющих на отчисление. 
В данных представлены демографические характеристики,
</span>
""", unsafe_allow_html=True)
st.markdown("""
<span style='color:#8B0000'>
детали поступления, академические показатели 
и социально-экономический контекст. 
Переменная Target показывает текущий статус студента: обучается, выпустился или отчислен.
</span>
""", unsafe_allow_html=True)

# 2. Создаём три колонки для графиков
col1, col2, col3 = st.columns(3)


with col1:
    # График 1: Распределение целевой переменной
    st.subheader("Распределение классов")
    fig1 = px.pie(
        names=['Отчислен', 'Выпустился', 'Обучается'], 
        values=[794, 2209, 1421],
        color_discrete_sequence=['#8B0000', '#CD5C5C', '#FF6347'],
        hole=0.3
    )
    fig1.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        textfont_color='white'
    )
    fig1.update_layout(
        width=400,
        height=400,
        margin=dict(l=50, r=50, b=50, t=50, pad=4),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # График важности признаков с русскими названиями
    st.subheader("Топ 10 важных признаков!!")
    feature_importance_ru = pd.DataFrame({
        'Признак': [
            'Дисциплины 2 сем (сдано)',
            'Оплата обучения актуальна',
            'Дисциплины 1 сем (зачислено)',
            'Дисциплины 1 сем (сдано)',
            'Дисциплины 1 сем (оценки)',
            'Стипендиат',
            'Дисциплины 2 сем (зачислено)',
            'Должник',
            'Дисциплины 2 сем (оценки)',
            'Направление подготовки',
            'Возраст при поступлении',
            'Дисциплины 2 сем (сред. балл)',
            'Возраст 25-30 лет',
            'Форма обучения',
            'Профессия матери'
        ],
        'Важность': [
            0.191129, 0.086025, 0.044757, 0.040146, 0.033854,
            0.030744, 0.030717, 0.030699, 0.026809, 0.023268,
            0.022266, 0.021778, 0.021505, 0.021502, 0.020989
        ]
    })

    # Создаем график только для топ-10 признаков
    top_features = feature_importance_ru.sort_values('Важность', ascending=False).head(10)

    fig2 = px.bar(
        top_features.sort_values('Важность', ascending=True),
        x='Важность',
        y='Признак',
        orientation='h',
        color='Важность',
        color_continuous_scale=['#FFC0CB', '#8B0000'],
        labels={'Важность': 'Значимость'},
        title='Топ-10 значимых факторов'
    )

    fig2.update_layout(
        yaxis={'categoryorder':'total ascending'},
        font=dict(size=10, color='#8B0000'),  # Уменьшен размер шрифта
        width=350,  # Еще немного уменьшил ширину
        height=400,
        margin=dict(l=120, r=20, b=50, t=80, pad=4),  # Подправлены отступы
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis_range=[0, 0.2]
    )
    
    st.plotly_chart(fig2, use_container_width=True)  # Включил адаптацию к ширине колонки

with col3:
    # График 3: Интерактивная матрица ошибок
    st.subheader("Матрица ошибок")
    
    cm = np.array([[42, 39, 274],
                  [29, 511, 12],
                  [81, 68, 50]])
    z = cm.tolist()
    x = ['Отчислен', 'Выпустился', 'Обучается']
    y = ['Отчислен', 'Выпустился', 'Обучается']
    
    fig3 = ff.create_annotated_heatmap(
        z, x=x, y=y, 
        colorscale=[[0, '#FFC0CB'], [0.5, '#CD5C5C'], [1, '#8B0000']],
        annotation_text=[[str(y) for y in x] for x in z],
        showscale=True
    )
    
    fig3.update_layout(
        width=400,
        height=400,
        margin=dict(l=50, r=50, b=50, t=50, pad=4),
        xaxis_title='Предсказанные классы',
        yaxis_title='Истинные классы',
        font=dict(color='#8B0000', size=12),
        xaxis=dict(side='bottom', tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12)),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig3, use_container_width=True)

# 3. Метрики модели
st.header("Результаты модели XGBoost")
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

with metrics_col1:
    st.metric("Accuracy (Точность)", "0.78", help="Общая точность модели")

with metrics_col2:
    st.metric("Recall (Полнота)", "0.40", help="Доля правильно предсказанных отчислений")

with metrics_col3:
    st.metric("Precision (Точность)", "0.81", help="Точность предсказаний выпускников")

with metrics_col4:
    st.metric("F1-score (F-мера)", "0.70", help="Среднее гармоническое precision и recall")