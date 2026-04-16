import streamlit as st 
import joblib
import pandas as pd
from model_utils import predict_expense, analyze_expense

#loead saved files
model=joblib.load("expense_model.pkl")
scaler=joblib.load("scaler.pkl")
encoder=joblib.load("encoder.pkl")

#updated city tier
city_tier_map = {
    "Mumbai": "Tier 1-Elite",
    "Bangalore": "Tier 1-Elite",
    "Delhi": "Tier 1-Elite",
    "Chennai": "Tier 1-Standard",
    "Hyderabad": "Tier 1-Standard",
    "Pune": "Tier 1-Standard",
    "Kochi": "Tier 2",
    "Bhopal": "Tier 2",
    "Indore": "Tier 2"
}

#title
st.title("Smart Expense Predictor(India)")

st.sidebar.header("Enter Your Details")
#user inputs
user_type = st.sidebar.selectbox("User Type", ["Working Professional", "Student/Intern"])

city = st.selectbox("City", list(city_tier_map.keys()))

tier = city_tier_map[city]

salary = st.sidebar.number_input("Monthly Salary (₹)", min_value=1000, max_value=1000000, step=1000)

family_size = st.sidebar.slider("Family Size", 1, 10)

lifestyle = st.sidebar.selectbox("Lifestyle", ["Low", "Medium", "High"])

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
    
    st.subheader(" Expense Breakdown")
    
    col1,col2=st.columns(2)
    
    with col1:
        st.metric("Rent",f"₹{int(result['Rent'])}")
        st.metric("Food", f"₹{int(result['Food'])}")
        st.metric("Transport", f"₹{int(result['Transport'])}")
    with col2:
        st.metric("Entertainment", f"₹{int(result['Entertainment'])}")
        st.metric("Other", f"₹{int(result['Other'])}")
        st.metric("Savings", f"₹{int(result['Savings'])}")
    
    st.subheader("Insights")
    
    for insight in result["Insights"]:
        st.warning(insight)
    
    import pandas as pd
    
    expense_data={
        "Category": ["Rent", "Food", "Transport", "Entertainment", "Other"],
        "Amount":[
            result["Rent"],
            result["Food"],
            result["Transport"],
            result["Entertainment"],
            result["Other"]
        ]
    }
    df_chart=pd.DataFrame(expense_data)
    
    st.subheader("Expense Distribution")
    st.bar_chart(df_chart.set_index("Category"))
    
    
    