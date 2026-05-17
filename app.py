import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Налаштування сторінки
st.set_page_config(page_title="Health Risk Predictor", layout="wide")

# Завантаження моделі
@st.cache_resource
def load_model():
    try:
        model = joblib.load('health_model_pipeline.pkl')
        return model
    except FileNotFoundError:
        st.error("Файл моделі 'health_model_pipeline.pkl' не знайдено. Будь ласка, завантажте його.")
        return None

pipeline = load_model()

st.title("Система оцінки потреби в госпіталізації")
st.markdown("""
Цей вебсервіс використовує модель машинного навчання для прогнозування ризику госпіталізації на основі введених клінічних та соціальних показників пацієнта.
""")

# Розділення інтерфейсу на колонки
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Вхідні дані пацієнта")
    
    # Група 1: Демографія та базові показники
    st.subheader("Демографія")
    age = st.slider("Вік", 18, 100, 50)
    gender = st.selectbox("Стать", ["Male", "Female"])
    ses = st.selectbox("Соціально-економічний статус (SES)", ["Low", "Medium", "High"])
    
    # Група 2: Медичні показники
    st.subheader("Медичні показники")
    temperature = st.slider("Температура тіла (°C)", 35.5, 41.0, 36.6, step=0.1)
    chronic = st.number_input("Кількість хронічних захворювань", 0, 10, 0)
    vaccination = st.radio("Статус вакцинації", ["Ні", "Так"])
    vaccination_status = 1 if vaccination == "Так" else 0
    immunity = st.selectbox("Рівень імунітету", ["Low", "Medium", "High"])
    symptoms = st.selectbox("Симптоми", ["None", "Mild", "Moderate", "Severe"])
    
    # Група 3: Соціальні фактори
    st.subheader("Соціальні фактори")
    social_activity = st.selectbox("Соціальна активність", ["Low", "Medium", "High"])
    
with col2:
    st.header("Результати прогнозування")
    
    if st.button("Розрахувати ризик", type="primary"):
        if pipeline is not None:
            # Формування словника з усіма необхідними ознаками для пайплайну
            # Використовуються введені дані, інші заповнюються типовими значеннями (медіана/мода)
            input_data = {
                'Age': age,
                'Gender': gender,
                'Location': 'Urban', # Типове значення
                'Ethnicity': 'Ethnicity1', # Типове значення
                'SES': ses,
                'Chronic_Conditions': chronic,
                'Vaccination_Status': vaccination_status,
                'Medical_History': 'None',
                'Immunity_Level': immunity,
                'Reported_Symptoms': symptoms,
                'Diagnosis': 'None',
                'Testing_Results': 'Negative',
                'Temperature': temperature,
                'AQI': 100, # Середнє значення
                'Humidity': 50.0, # Середнє значення
                'Population_Density': 'Medium',
                'Travel_History': 'No Travel',
                'Social_Activity': social_activity,
                'Compliance_with_Health_Guidelines': 1,
                'Vaccination_Hesitancy': 'No',
                'Transmission_Rate': 1.5, # Середнє значення
                'Mortality_Rate': 0.02, # Середнє значення
                'Case_Fatality_Ratio': 0.05, # Середнє значення
                'Hospitalization_Rate': 'Medium',
                'Hospital_Capacity': 'Available',
                'Healthcare_Personnel_Availability': 'Adequate',
                'Resource_Utilization': 50.0, # Середнє значення
                'Daily_New_Cases': 50,
                'Outbreak_Status': 'No Outbreak',
                'Infection_Risk_Level': 'Medium Risk',
                'Disease_Severity': 'Mild',
                'Risk_Index': 1.5 * 0.02 # Сконструйована ознака з Розділу 1
            }
            
            # Перетворення у DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Прогноз
            try:
                prediction = pipeline.predict(input_df)[0]
                proba = pipeline.predict_proba(input_df)[0][1]
                
                # Відображення результату
                st.metric("Ймовірність госпіталізації", f"{proba * 100:.1f}%")
                
                if proba < 0.3:
                    st.success("Профіль ризику: Низький. Госпіталізація не потрібна.")
                elif proba < 0.7:
                    st.warning("Профіль ризику: Середній. Рекомендовано нагляд лікаря.")
                else:
                    st.error("Профіль ризику: Високий. Потребує негайної госпіталізації!")
                
                # Інтерпретація SHAP
                st.subheader("Пояснення моделі (Внесок ознак)")
                with st.spinner('Обчислення важливості ознак...'):
                    # Отримання класифікатора та оброблених даних
                    clf = pipeline.named_steps['clf']
                    prep = pipeline.named_steps['prep']
                    
                    X_proc = prep.transform(input_df)
                    num_features = prep.transformers_[0][2]
                    cat_encoder = prep.named_transformers_['cat'].named_steps['onehot']
                    cat_features_in = prep.transformers_[1][2]
                    cat_feature_names = cat_encoder.get_feature_names_out(cat_features_in).tolist()
                    all_feature_names = num_features + cat_feature_names
                    
                    X_proc_df = pd.DataFrame(X_proc, columns=all_feature_names)
                    
                    explainer = shap.TreeExplainer(clf)
                    shap_values = explainer.shap_values(X_proc_df, check_additivity=False)
                    
                    if isinstance(shap_values, list):
                        shap_val = shap_values[1][0]
                    else:
                        shap_val = shap_values[0]
                    
                    # Побудова графіка
                    fig = shap.waterfall_plot(shap.Explanation(values=shap_val, 
                                                              base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
                                                              data=X_proc_df.iloc[0], 
                                                              feature_names=all_feature_names), 
                                              show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()
                    
            except Exception as e:
                st.error(f"Виникла помилка при обробці даних: {e}")