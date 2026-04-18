# Smart Expense Predictor (India)

A machine learning-based web app that predicts monthly expense breakdown for Indian users based on salary, city, lifestyle, and family size.

It also provides financial insights to help users manage spending and improve savings.

##  Live Demo
https://your-app.streamlit.app

##  Features

- Predicts monthly expenses (Rent, Food, Transport, Entertainment, Other)
- Supports multiple Indian cities
- Automatically assigns city tier based on location
- Adjusts spending based on lifestyle (Low / Medium / High)
- Provides financial insights and warnings
- Handles unrealistic predictions using rule-based corrections

##  Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

##  How It Works

1. User inputs salary, city, lifestyle, and family size
2. Data is preprocessed using encoding and scaling
3. A trained machine learning model predicts expense categories
4. A rule-based system adjusts predictions and generates insights
5. Results are displayed through an interactive Streamlit UI

## 📁 Project Structure

smart-expense-predictor/

── app.py                # Streamlit UI
── model_utils.py        # Prediction + analysis logic
── expense_model.pkl     # Trained model
── scaler.pkl            # Scaler
── encoder.pkl           # Encoder
── requirements.txt      # Dependencies

##  Limitations

- Model accuracy depends on dataset quality
- Limited real-world data for extreme salary ranges
- Lifestyle behavior is partially rule-based, not fully learned

##  Future Improvements

- Use real-world financial datasets
- Improve model accuracy for edge cases
- Add expense tracking over time
- Integrate with banking APIs
