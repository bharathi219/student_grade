import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the feature names used during training
model = joblib.load("random_forest_model.pkl")
features_used = joblib.load("features_used.pkl")

st.title("🎓 Student Final Grade Predictor")
st.markdown("Enter the student's academic details below to predict their final exam score (G3).")

# Collecting inputs from the user
G1 = st.slider("G1 (First Period Grade)", 0, 20, 10)
G2 = st.slider("G2 (Second Period Grade)", 0, 20, 10)
studytime = st.selectbox("Study Time (1 = <2 hrs, 4 = >10 hrs)", [1, 2, 3, 4])
failures = st.number_input("Number of Past Class Failures", 0, 4, 0)
absences = st.number_input("Number of Absences", 0, 100, 0)
health = st.slider("Health Status (1 = Poor, 5 = Excellent)", 1, 5, 3)
internet_yes = st.radio("Internet Access at Home", ["Yes", "No"]) == "Yes"
famsup_yes = st.radio("Family Educational Support", ["Yes", "No"]) == "Yes"

# Convert boolean inputs to integers
internet_yes = int(internet_yes)
famsup_yes = int(famsup_yes)

# Build a dictionary of available user inputs
user_input = {
    'G1': G1,
    'G2': G2,
    'studytime': studytime,
    'failures': failures,
    'absences': absences,
    'health': health,
    'internet_yes': internet_yes,
    'famsup_yes': famsup_yes
}

# Create a full input row with all expected features
input_data = pd.DataFrame([{
    feature: user_input.get(feature, 0) for feature in features_used
}])

# Prediction
if st.button("Predict Final Grade"):
    prediction = model.predict(input_data)[0]
    st.success(f"🎯 Predicted Final Grade (G3): **{round(prediction, 2)}** out of 20")
