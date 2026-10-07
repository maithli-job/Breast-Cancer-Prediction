import streamlit as st
import pandas as pd
import joblib
# LOAD MODEL FILES

model = joblib.load("breast_cancer_best_model.pkl")

scaler = joblib.load("breast_cancer_scaler.pkl")

selected_features = joblib.load(
    "breast_cancer_features.pkl"
)

# PAGE SETTINGS
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)
# TITLE
st.title("🩺 Breast Cancer Tumor Classification")

st.write(
    "Enter the tumor measurements below to predict "
    "whether the tumor is Benign or Malignant."
)

st.info(
    "This application is for educational purposes "
    "and is not a medical diagnosis."
)

# INPUTS
st.subheader("Enter Tumor Measurements")

col1, col2 = st.columns(2)


with col1:

    radius_mean = st.number_input(
        "Radius Mean",
        min_value=0.0,
        value=14.0,
        step=0.1
    )

    perimeter_mean = st.number_input(
        "Perimeter Mean",
        min_value=0.0,
        value=90.0,
        step=0.1
    )

    area_mean = st.number_input(
        "Area Mean",
        min_value=0.0,
        value=650.0,
        step=1.0
    )

    concavity_mean = st.number_input(
        "Concavity Mean",
        min_value=0.0,
        value=0.09,
        step=0.01
    )

    concave_points_mean = st.number_input(
        "Concave Points Mean",
        min_value=0.0,
        value=0.05,
        step=0.01
    )


with col2:

    radius_worst = st.number_input(
        "Radius Worst",
        min_value=0.0,
        value=16.0,
        step=0.1
    )

    perimeter_worst = st.number_input(
        "Perimeter Worst",
        min_value=0.0,
        value=105.0,
        step=0.1
    )

    area_worst = st.number_input(
        "Area Worst",
        min_value=0.0,
        value=800.0,
        step=1.0
    )

    concavity_worst = st.number_input(
        "Concavity Worst",
        min_value=0.0,
        value=0.25,
        step=0.01
    )

    concave_points_worst = st.number_input(
        "Concave Points Worst",
        min_value=0.0,
        value=0.10,
        step=0.01
    )

# PREDICTION
if st.button(
    " Predict Tumor",
    use_container_width=True
):

    input_data = pd.DataFrame([{

        "radius_mean": radius_mean,

        "perimeter_mean": perimeter_mean,

        "area_mean": area_mean,

        "concavity_mean": concavity_mean,

        "concave points_mean": concave_points_mean,

        "radius_worst": radius_worst,

        "perimeter_worst": perimeter_worst,

        "area_worst": area_worst,

        "concavity_worst": concavity_worst,

        "concave points_worst": concave_points_worst

    }])


    # Keep exactly the same columns
    # and same order used during training

    input_data = input_data[
        selected_features
    ]


    # Scale the input

    input_scaled = scaler.transform(
        input_data
    )


    # Prediction

    prediction = model.predict(
        input_scaled
    )[0]


    # Probability

    probability = model.predict_proba(
        input_scaled
    )[0]


    benign_probability = (
        probability[0] * 100
    )

    malignant_probability = (
        probability[1] * 100
    )
    # RESULT
    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            " MALIGNANT TUMOR"
        )

        st.metric(
            "Malignant Probability",
            f"{malignant_probability:.2f}%"
        )

    else:

        st.success(
            " BENIGN TUMOR"
        )

        st.metric(
            "Benign Probability",
            f"{benign_probability:.2f}%"
        )
    # PROBABILITY TABLE

    st.subheader(
        "Prediction Probabilities"
    )

    probability_df = pd.DataFrame({

        "Class": [
            "Benign",
            "Malignant"
        ],

        "Probability": [
            f"{benign_probability:.2f}%",
            f"{malignant_probability:.2f}%"
        ]

    })

    st.table(
        probability_df
    )

