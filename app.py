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
        st.error("Файл моделі 'health_model_pipeline.pkl' не знайдено.")
        return None

pipeline = load_model()

st.title("Система оцінки потреби в госпіталізації")
st.markdown("Цей вебсервіс використовує модель машинного навчання для прогнозування ризику госпіталізації.")

# Розділення інтерфейсу на колонки
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Вхідні дані пацієнта")
    
    st.subheader("Демографія")
    # Додано унікальні ключі для кожного елемента
    age = st.slider("Вік", 18, 100, 50, key="input_age")
    gender = st.selectbox("Стать", ["Male", "Female"], key="input_gender")
    ses = st.selectbox("Соціально-економічний статус (SES)", ["Low", "Medium", "High"], key="input_ses")
    
    st.subheader("Медичні показники")
    temperature = st.slider("Температура тіла (°C)", 35.5, 41.0, 36.6, step=0.1, key="input_temp")
    chronic = st.number_input("Кількість хронічних захворювань", 0, 10, 0, key="input_chronic")
    vaccination = st.radio("Статус вакцинації", ["Ні", "Так"], key="input_vax")
    vaccination_status = 1 if vaccination == "Так" else 0
    immunity = st.selectbox("Рівень імунітету", ["Low", "Medium", "High"], key="input_immunity")
    symptoms = st.selectbox("Повідомлені симптоми", ["None", "Mild", "Moderate", "Severe"], key="input_symptoms")
    
    disease_severity = st.selectbox("Тяжкість захворювання", ["Mild", "Moderate", "Severe"], index=1, key="input_severity")
    diagnosis = st.selectbox("Поточний діагноз", ["None", "Disease1", "Disease2", "Disease3"], key="input_diagnosis")
    risk_level = st.selectbox("Рівень інфекційного ризику", ["Low Risk", "Medium Risk", "High Risk"], index=1, key="input_risk")
    
    st.subheader("Соціальні фактори")
    social_activity = st.selectbox("Соціальна активність", ["Low", "Medium", "High"], key="input_social")

with col2:
    st.header("Результати прогнозування")
    
    if st.button("Розрахувати ризик", type="primary", key="calc_btn"):
        if pipeline is not None:
            input_data = {
                'Age': age,
                'Gender': gender,
                'Location': 'Urban',
                'Ethnicity': 'Ethnicity1',
                'SES': ses,
                'Chronic_Conditions': chronic,
                'Vaccination_Status': vaccination_status,
                'Medical_History': 'None',
                'Immunity_Level': immunity,
                'Reported_Symptoms': symptoms,
                'Diagnosis': diagnosis, 
                'Testing_Results': 'Negative',
                'Temperature': temperature,
                'AQI': 100,
                'Humidity': 50.0,
                'Population_Density': 'Medium',
                'Travel_History': 'No Travel',
                'Social_Activity': social_activity,
                'Compliance_with_Health_Guidelines': 1,
                'Vaccination_Hesitancy': 'No',
                'Transmission_Rate': 1.5,
                'Mortality_Rate': 0.02,
                'Case_Fatality_Ratio': 0.05,
                'Hospitalization_Rate': 'Medium',
                'Hospital_Capacity': 'Available',
                'Healthcare_Personnel_Availability': 'Adequate',
                'Resource_Utilization': 50.0,
                'Daily_New_Cases': 50,
                'Outbreak_Status': 'No Outbreak',
                'Infection_Risk_Level': risk_level, 
                'Disease_Severity': disease_severity, 
                'Risk_Index': 1.5 * 0.02 
            }
            
            input_df = pd.DataFrame([input_data])
            
            try:
                prediction = pipeline.predict(input_df)[0]
                proba = pipeline.predict_proba(input_df)[0][1]
                
                st.metric("Ймовірність госпіталізації", f"{proba * 100:.1f}%")
                
                if proba < 0.3:
                    st.success("Профіль ризику: Низький. Госпіталізація не потрібна.")
                elif proba < 0.7:
                    st.warning("Профіль ризику: Середній. Рекомендовано нагляд лікаря.")
                else:
                    st.error("Профіль ризику: Високий. Потребує негайної госпіталізації!")
                
                st.subheader("Пояснення моделі (Внесок ознак)")
                with st.spinner('Обчислення важливості ознак...'):
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
                    
                    fig = shap.waterfall_plot(shap.Explanation(values=shap_val, 
                                                              base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
                                                              data=X_proc_df.iloc[0], 
                                                              feature_names=all_feature_names), 
                                              show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()
                    
            except Exception as e:
                st.error(f"Виникла помилка при обробці даних: {e}")
