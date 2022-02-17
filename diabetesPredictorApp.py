import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import seaborn as sns
import time

st.title("Diabetes Predictor App")
st.write("Pur your BMI, Glucose Level, Age and BP reading to determine if you have diabetes")

@st.cache(suppress_st_warning=True)
def load_csv_data(df):
    df = pd.read_csv("diabetes_classification.csv")
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        latest_iteration.text(f'Loading CSV Data... {i+1}%')
        bar.progress(i+1)
        time.sleep(0.01)
    bar.empty()
    latest_iteration.text('')
    return df

df = load_csv_data("diabetes_classification.csv")
columns = ["Age", "BMI", "Glucose", "BloodPressure"]

st.sidebar.title("Parameters")
st.sidebar.write("Tweak to change see your result")

age = st.sidebar.slider("Age", 0, 100, 50)

bmi = st.sidebar.slider("BMI", 15, 40, 33)

glucose = st.sidebar.slider("Glucose", 0, 200, 148)

bp = st.sidebar.slider("Blood Pressure", 0, 200, 72)

st.subheader("Predictions")

filename = 'diabetes_model.sav'
loaded_model = joblib.load(filename)

prediction = round(loaded_model.predict([[age, bmi, glucose, bp]])[0])

# st.write(f"Risk of diabetes: {prediction}")

if prediction == 0:
    risk_status = "Low"
else:
    risk_status = "High"
    
st.write(f"Risk to Diabetes: {risk_status}")


if st.sidebar.checkbox('Show/Hide dataframe'):
    st.write(df.head(20))

if st.checkbox('Show line chart'):
    st.write("Line Chart")
    st.line_chart(df[columns])
    
if st.checkbox('Show bar chart'):
    st.write("Line Chart")
    st.bar_chart(df[columns])

if st.checkbox('Show area chart'):
    st.write("Area Chart")
    st.area_chart(df[columns])
