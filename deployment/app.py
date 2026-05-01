import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="priyankaprakasan/tourism-model", filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Wellness Tourism Package Prediction
st.title("Wellness Tourism Package Prediction App")
st.write("This app predicts whether a customer will purchase the Wellness Tourism Package.")
st.write("Please enter the customer details to get a prediction.")

# Collect user input based on tourism.csv features
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
NumberOfTrips = st.number_input("Number of Trips Annually", min_value=0, max_value=20, value=1)
Passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
OwnCar = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=15)
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=30000.0)
TypeofContact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Director"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'ProductPitched': ProductPitched
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase the package" if prediction == 1 else "not purchase the package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
