import pandas as pd
import numpy as np

def predict_expense(user_input,model,encoder,scaler):
      #convert input to DF
  input_df=pd.DataFrame([user_input])

  #separate categorical and numerical
  categorical_cols=['user_type','city','tier','lifestyle']
  numerical_cols=['salary','family_size']

  #encode categorical data
  encoded=encoder.transform(input_df[categorical_cols])
  encoded_df=pd.DataFrame(encoded,columns=encoder.get_feature_names_out(categorical_cols))

  #combine with numerical
  final_input=pd.concat([encoded_df,input_df[numerical_cols].reset_index(drop=True)],axis=1)

  #scale input
  scaleed_input=scaler.transform(final_input)

  #predict
  prediction=model.predict(scaleed_input)

  #convert to readable format
  output={
      'Rent':prediction[0][0],
      'Food':prediction[0][1],
      'Transport':prediction[0][2],
      'Entertainment':prediction[0][3],
      'Other':prediction[0][4]
  }
  return output



#adding suggestions
def analyze_expense(user_input, prediction):
    
    #extract inputs
    salary = user_input.get('salary', 0)
    family_size = user_input.get('family_size', 1)
    lifestyle = user_input.get('lifestyle', 'Medium')

    #extract predictions
    rent = prediction.get('Rent', 0)
    food = prediction.get('Food', 0)
    transport = prediction.get('Transport', 0)
    entertainment = prediction.get('Entertainment', 0)
    other = prediction.get('Other', 0)

    #lifestyle adjustment
    if lifestyle == "High":
        food *= 1.2
        entertainment *= 1.3
        transport *= 1.1

    elif lifestyle == "Low":
        food *= 0.8
        entertainment *= 0.7
        transport *= 0.9

    #cap unrealistic food spending
    if food > 0.35 * salary:
        food = (0.25 + 0.02 * family_size) * salary

    #calculate totals
    total_expense = rent + food + transport + entertainment + other
    savings = salary - total_expense

    #insights
    insights = []

    if rent > 0.4 * salary:
        insights.append("Rent is too high")

    if food > 0.3 * salary:
        insights.append("Food spending is high")

    if transport > 0.2 * salary:
        insights.append("Transport cost is high")

    if savings < 0:
        insights.append("You are overspending!")

    elif savings < 0.1 * salary:
        insights.append("Low savings")

    elif savings > 0.3 * salary:
        insights.append("Good savings habit")

    #final result
    return {
        "Rent": rent,
        "Food": food,
        "Transport": transport,
        "Entertainment": entertainment,
        "Other": other,
        "Total Expense": total_expense,
        "Savings": savings,
        "Insights": insights
    }

#user input
# prediction=predict_expense(user,model,encoder,scaler)
# final_output=analyze_expense(user,prediction)
# print(final_output)
