import streamlit as st 
import joblib
import pandas as pd
from model_utils import predict_expense, analyze_expense

#loead saved files
model=joblib.load("expense_model.pkl")
scaler=joblib.load("scaler.pkl")
encoder=joblib.load("encoder.pkl")

#title
st.title("Smart Expense Predictor(India)")

#user inputs
user_type=st.selectbox("User Type",["Working Professional", "Student/Intern"])
city=st.selectbox("City",["Mumbai", "Chennai", "Bangalore", "Kochi"])
tier = st.selectbox("City Tier", ["Tier 1-Elite", "Tier 1-Standard", "Tier 2"])
salary=st.number_input("Monthly Salary(₹)",min_value=1000,max_value=1000000,step=1000)
family_size=st.slider("Family Size",1,10)
lifestyle=st.selectbox("Lifestyle",["Low","Medium", "High"])

#predict button
if st.button("Predict Expenses"):
    
    user={
        'user_type': user_type,
        'city': city,
        'tier': tier,
        'salary': salary,
        'family_size': family_size,
        'lifestyle': lifestyle
    }
    prediction=predict_expense(user,model,encoder,scaler)
    result=analyze_expense(user,prediction)
    
    st.subheader("Predict Expenses")
    #st.write(result)
    st.metric("Rent", f"₹{int(result['Rent'])}")
    st.metric("Food", f"₹{int(result['Food'])}")
    