import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction")

st.write(
    "Enter the student's study, sleep, stress and previous exam details "
    "to predict the performance score."
)


@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model(
        "student_performance_ann.h5"
    )

    with open("student_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_artifacts()


st.subheader("Enter Student Details")


sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)


stress_level = st.number_input(
    "Stress Level",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)


previous_exam_scores = st.number_input(
    "Previous Exam Scores",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)


study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)


if st.button("Predict Performance"):

    input_data = pd.DataFrame(
        [[
            sleep_hours,
            stress_level,
            previous_exam_scores,
            study_hours
        ]],
        columns=[
            "Sleep_Hours",
            "Stress_Level",
            "Previous_Exam_Scores",
            "Study_Hours"
        ]
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(
        input_scaled,
        verbose=0
    )

    predicted_score = float(prediction[0][0])

    st.subheader("Prediction Result")

    st.success(
        f"Predicted Performance Score: {predicted_score:.2f}"
    )