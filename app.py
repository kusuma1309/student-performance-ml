import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("student_model.pkl", "rb"))

st.title("🎓 Student Performance Prediction")

st.write("Predict student's math score based on academic features.")


# User Inputs
gender = st.selectbox("Gender", ["Female", "Male"])

race = st.selectbox(
    "Race / Ethnicity",
    ["Group A", "Group B", "Group C", "Group D", "Group E"]
)

parent_edu = st.slider("Parental Education Level (Encoded)", 0, 5)

lunch = st.selectbox("Lunch Type", ["Standard", "Free/Reduced"])

test_prep = st.selectbox("Test Preparation", ["None", "Completed"])

reading_score = st.slider("Reading Score", 0, 100)

writing_score = st.slider("Writing Score", 0, 100)


# Encoding values (same as training)
gender = 1 if gender == "Male" else 0

race_map = {
    "Group A":0,
    "Group B":1,
    "Group C":2,
    "Group D":3,
    "Group E":4
}

race = race_map[race]

lunch = 1 if lunch == "Standard" else 0

test_prep = 1 if test_prep == "Completed" else 0


# Feature Engineering
total_score = reading_score + writing_score


# Prediction
if st.button("Predict Math Score"):

    data = np.array([[gender, race, parent_edu, lunch, test_prep,
                      reading_score, writing_score, total_score]])

    prediction = model.predict(data)

    st.success(f"Predicted Math Score: {prediction[0]:.2f}")