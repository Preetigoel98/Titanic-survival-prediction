import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# ------------------------------
# Load trained model
# ------------------------------
# Assuming you saved model earlier using pickle
# Example: pickle.dump(log_model, open("titanic_model.pkl", "wb"))
model = pickle.load(open("titanic_model.pkl", "rb"))

st.title("🚢 Titanic Survival Prediction App")

st.write("Enter passenger details to predict survival probability.")

# ------------------------------
# User Inputs
# ------------------------------
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Fare Price", 0.0, 600.0, 32.0)

embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode inputs just like training
sex = 1 if sex == "Male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Create dataframe for model input
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked_Q": [embarked_Q],
    "Embarked_S": [embarked_S]
})

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Passenger Survived (Probability: {prob:.2f})")
    else:
        st.error(f"❌ Passenger Did Not Survive (Probability: {prob:.2f})")
