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
st.markdown("Цей вебсервіс використовує модель машинного навчання для прогнозування ризику госпіталізації на основі введених клінічних та соціальних показників пацієнта.")

# Розділення інтерфейсу на колонки
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Вхідні дані пацієнта")
    
    st.subheader("Демографія")
    age = st.slider("Вік", 18, 100, 50, key="input_age")
    gender = st.selectbox("Стать", ["Male", "Female"], key="input_gender")
    ses = st.selectbox("Соціально-економічний статус (SES)", ["Low", "Medium", "High"], key="input_ses")
    
    st.subheader("Медичні показники")
    chronic = st.number_input("Кількість хронічних захворювань", 0, 10, 0, key="input_chronic")
    vaccination = st.radio("Статус вакцинації", ["Ні", "Так"], key="input_vax")
    vaccination_status = 1 if vaccination == "Так" else 0
    immunity = st.selectbox("Рівень імунітету", ["Low", "Medium", "High"], key="input_immunity")
    
    st.subheader("Клінічний стан та тести")
    testing_results = st.selectbox("Результати тесту", ["Negative", "Positive"], index=1, key="input_test")
    symptoms = st.selectbox("Повідомлені симптоми", ["None", "Mild", "Moderate", "Severe"], index=3, key="input_symptoms")
    disease_severity = st.selectbox("Тяжкість захворювання", ["Mild", "Moderate", "Severe"], index=2, key="input_severity")
    diagnosis = st.selectbox("Поточний діагноз", ["None", "Disease1", "Disease2", "Disease3"], index=1, key="input_diagnosis")
    risk_level = st.selectbox("Рівень інфекційного ризику", ["Low Risk", "Medium Risk", "High Risk"], index=2, key="input_risk")
    
    st.subheader("Екологічні та епідеміологічні фактори")
    temperature = st.slider("Температура середовища (°C)", -15.0, 50.0, 30.0, step=0.1, key="input_temp")
    humidity = st.slider("Вологість (%)", 0.0, 100.0, 50.0, key="input_hum")
    aqi = st.slider("Індекс якості повітря (AQI)", 0, 300, 50, key="input_aqi")
    transmission = st.slider("Швидкість передачі інфекції", 0.1, 5.0, 1.74, step=0.1, key="input_trans")
    daily_cases = st.slider("Нові випадки за день", 0, 100, 20, key="input_cases")
    
    st.subheader("Соціальні фактори")
    social_activity = st.selectbox("Соціальна активність", ["Low", "Medium", "High"], key="input_social")

with col2:
    st.header("Результати прогнозування")
    
    if st.button("Розрахувати ризик", type="primary", key="calc_btn"):
        if pipeline is not None:
            # Словник із точними медіанами та модами для нейтралізації фонових ознак
          input_data = {
                'Age': age,
                'Gender': gender,
                'Location': 'Urban',
                'Ethnicity': 'Ethnicity1',
                'SES': ses,
                'Chronic_Conditions': chronic,
                'Vaccination_Status': vaccination_status,
                'Medical_History': 'Past Illness',
                'Immunity_Level': immunity,
                'Reported_Symptoms': symptoms,
                'Diagnosis': diagnosis, 
                'Testing_Results': testing_results, 
                'Temperature': temperature,
                'AQI': aqi, # ТЕПЕР З ІНТЕРФЕЙСУ
                'Humidity': humidity, # ТЕПЕР З ІНТЕРФЕЙСУ
                'Population_Density': 'Medium',
                'Travel_History': 'No Travel',
                'Social_Activity': social_activity,
                'Compliance_with_Health_Guidelines': 1,
                'Vaccination_Hesitancy': 'No',
                'Transmission_Rate': transmission, # ТЕПЕР З ІНТЕРФЕЙСУ
                'Mortality_Rate': 0.025, 
                'Case_Fatality_Ratio': 0.049, 
                'Hospitalization_Rate': 'Low', 
                'Hospital_Capacity': 'Available', 
                'Healthcare_Personnel_Availability': 'Adequate', 
                'Resource_Utilization': 50.1, 
                'Daily_New_Cases': daily_cases, # ТЕПЕР З ІНТЕРФЕЙСУ
                'Outbreak_Status': 'No Outbreak', 
                'Infection_Risk_Level': risk_level, 
                'Disease_Severity': disease_severity, 
                'Risk_Index': transmission * 0.025 # Оновлено
            }
            
            # Перетворення у DataFrame
            input_df = pd.DataFrame([input_data])
            
            try:
                # Прогноз
                prediction = pipeline.predict(input_df)[0]
                proba = pipeline.predict_proba(input_df)[0][1]
                
                # Відображення результату
                st.metric("Ймовірність потреби в госпіталізації", f"{proba * 100:.1f}%")
                
                if proba < 0.3:
                    st.success("Профіль ризику: Низький. Госпіталізація не потрібна.")
                elif proba < 0.7:
                    st.warning("Профіль ризику: Середній. Рекомендовано нагляд лікаря.")
                else:
                    st.error("Профіль ризику: Високий. Потребує негайної госпіталізації!")
                
                # Інтерпретація SHAP
                st.subheader("Пояснення моделі (Внесок ознак)")
                with st.spinner('Обчислення важливості ознак. Зачекайте...'):
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
