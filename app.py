import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model

# Load the trained model and training data
model = load_model('best_kidney_disease_model')
train_data = joblib.load('kidney_train_data.pkl')

# Convert training data to numeric and handle any potential NaNs
train_data = train_data.apply(pd.to_numeric, errors='coerce')
train_data.fillna(train_data.median(), inplace=True)

# Function to get user input
def get_user_input():
    age = st.number_input('Age', min_value=0, max_value=120, value=0)
    bp = st.number_input('Blood Pressure', min_value=0, max_value=200, value=0)
    sg = st.number_input('Specific Gravity', min_value=0.0, max_value=1.5, value=1.0)
    al = st.number_input('Albumin', min_value=0, max_value=5, value=0)
    bgr = st.number_input('Blood Glucose Random', min_value=0, max_value=500, value=0)
    bu = st.number_input('Blood Urea', min_value=0, max_value=300, value=0)
    sc = st.number_input('Serum Creatinine', min_value=0.0, max_value=50.0, value=0.0)
    sod = st.number_input('Sodium', min_value=0.0, max_value=200.0, value=0.0)
    pot = st.number_input('Potassium', min_value=0.0, max_value=10.0, value=0.0)
    hemo = st.number_input('Hemoglobin', min_value=0.0, max_value=20.0, value=0.0)
    wc = st.number_input('White Blood Cell Count', min_value=0, max_value=100000, value=0)
    htn = st.selectbox('Hypertension', [0, 1], format_func=lambda x: ["No", "Yes"][x])
    dm = st.selectbox('Diabetes Mellitus', [0, 1], format_func=lambda x: ["No", "Yes"][x])
    appet = st.selectbox('Appetite', [0, 1], format_func=lambda x: ["Poor", "Good"][x])
    rbc = st.selectbox('Red Blood Cells', [0, 1], format_func=lambda x: ["Normal", "Abnormal"][x])
    ane = st.selectbox('Anemia', [0, 1], format_func=lambda x: ["No", "Yes"][x])

    user_data = {
        'age': age,
        'bp': bp,
        'sg': sg,
        'al': al,
        'bgr': bgr,
        'bu': bu,
        'sc': sc,
        'sod': sod,
        'pot': pot,
        'hemo': hemo,
        'wc': wc,
        'htn': htn,
        'dm': dm,
        'appet': appet,
        'rbc': rbc,
        'ane': ane
    }

    # Ensure that the DataFrame only includes features used during training
    feature_columns = ['age', 'bp', 'sg', 'al', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'wc', 'htn', 'dm', 'appet', 'rbc', 'ane']
    user_input_df = pd.DataFrame(user_data, index=[0])
    user_input_df = user_input_df[feature_columns]  # Keep only the relevant features

    # Convert to numeric to ensure compatibility
    user_input_df = user_input_df.apply(pd.to_numeric, errors='coerce')
    user_input_df.fillna(user_input_df.median(), inplace=True)
    
    return user_input_df

# Prediction Page
def prediction():
    st.title("Kidney Disease Prediction")
    user_input = get_user_input()

    if st.button("Predict"):
        prediction_result = predict_model(model, data=user_input)

        if 'prediction_label' in prediction_result.columns and 'prediction_score' in prediction_result.columns:
            predicted_class = int(prediction_result['prediction_label'].iloc[0])
            predicted_prob = prediction_result['prediction_score'].iloc[0]

            st.write("Prediction Result:")
            st.write(f"Predicted Class: {'CKD' if predicted_class == 1 else 'Non-CKD'}")
            st.write(f"Probability: {predicted_prob:.4f}")

            # Model Explainability using SHAP
            st.subheader("Model Explainability (SHAP)")
            shap.initjs()

            # Explaining the model's predictions using SHAP
            feature_columns = [col for col in train_data.columns if col != 'classification']
            explainer = shap.Explainer(model.predict, train_data[feature_columns])
            shap_values = explainer(user_input)

            # Feature Importance Plot
            st.subheader("Feature Importance")
            st.write("""
                The Feature Importance plot shows the most significant factors contributing to the prediction.
                The higher the bar, the more influence that feature has on the model's prediction.
            """)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, user_input, plot_type="bar", show=False)
            st.pyplot(fig)

            # SHAP Summary Plot
            st.subheader("SHAP Summary Plot")
            st.write("""
                The SHAP Summary plot provides a visualization of the impact of each feature on the model's output.
                Each dot represents a feature's impact on the prediction for a specific instance.
                The color represents the value of the feature (red for high, blue for low).
            """)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, user_input, show=False)
            st.pyplot(fig)

            # SHAP Force Plot
            st.subheader("SHAP Force Plot")
            st.write("""
                The SHAP Force plot illustrates the impact of each feature on the model's prediction for a single instance.
                Features pushing the prediction towards a higher value (indicating CKD) are shown in red,
                while those pushing towards a lower value (indicating Non-CKD) are in blue.
            """)
            st_shap(shap.force_plot(shap_values.base_values[0], shap_values.values[0], user_input))

            st.subheader("Detailed SHAP Explanation")
            st.write("""
                SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance.
                The SHAP value of each feature for a given prediction represents how much that feature
                contributes to the difference between the actual prediction and the average prediction for the dataset.
            """)

            for i, feature in enumerate(user_input.columns):
                st.write(f"**{feature}**: SHAP value = {shap_values.values[0][i]:.4f}")

            st.write("""
                - Positive SHAP values indicate that the feature contributes to predicting a higher probability of CKD.
                - Negative SHAP values indicate that the feature contributes to predicting a lower probability of CKD.
                - The magnitude of the SHAP value shows the strength of the contribution.
            """)

        else:
            st.error("Error: Prediction did not return 'prediction_label' or 'prediction_score' columns.")

# Helper function to display SHAP force plot in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# Main function
def main():
    prediction()

if __name__ == "__main__":
    main()
