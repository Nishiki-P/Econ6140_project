import streamlit as st
import numpy as np
import joblib

# Load the fitted pipeline from file
pipe = joblib.load("logit_pipeline.pkl")  # This pipeline includes both scaler and logistic regression

st.title("Remote Work Probability Predictor (COVID Period)")

st.write("""
This app estimates the probability that an individual was working remotely due to COVID-19  
(from May 2020 to September 2022), based on selected demographic and geographic features.
""")

# User inputs for the features
month_index = st.slider("Month index (e.g., 0 = May 2020)", 0, 30, 0)
male = st.radio("Gender", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
married = st.radio("Marital Status", options=[1, 0], format_func=lambda x: "Married" if x == 1 else "Not Married")
white = st.radio("Race", options=[1, 0], format_func=lambda x: "White" if x == 1 else "Non-White")
college_grad = st.radio("Education", options=[1, 0], format_func=lambda x: "College Graduate" if x == 1 else "Not a Graduate")
nrc = st.radio("Non-recursive Cognitive", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
nrm = st.radio("Non-recursive Manual", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
metro_flag = st.radio("Lives in Metro Area", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# On button click, predict the probability
if st.button("Predict"):
    X_input = np.array([[month_index, male, married, white,
                         college_grad, nrc, nrm, metro_flag]])
    probability = pipe.predict_proba(X_input)[0, 1]
    
    st.subheader("Predicted Probability")
    st.write(f"The estimated probability of remote work due to COVID-19 is **{probability:.2%}**.")
