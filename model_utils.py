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
def analyze_expense(user_input,prediction):

  salary=user_input['salary']
  family_size = user_input.get('family_size', 1)

  rent=prediction['Rent']
  food=prediction['Food']
  transport=prediction['Transport']
  entertainment=prediction['Entertainment']
  other=prediction['Other']
  lifestyle=user_input['lifestyle']
  
  #adjust spending based on lifestyle
  if lifestyle == "High":
        food *=1.2
        entertainment*=1.3
        transport*=1.1
  elif lifestyle == "Low":
        food *= 0.8
        entertainment *= 0.7
        transport *= 0.9
        
  # Cap unrealistic food spending
  if food > 0.35 * salary:
        food = (0.25 + 0.02 * family_size) * salary
          
  total_expenses=rent+food+transport+entertainment+other
  savings=salary-total_expenses

  insights=[]

  #rule 1: high rent
  if rent > 0.4*salary:
    insights.append("Rent is to heigh compared to salary")
  #rule 2: high food expense
  if food > 0.3*salary:
    insights.append("Food spending is high")
  #rule 3: high transport
  if transport > 0.2*salary:
    insights.append("Transport cost seems is high")
    transport= 0.15*salary
  #rule 4: low savings
  if savings < 0.1*salary:
    insights.append("Low savings, Risky")
  #rule 5: negative savings
  if savings < 0:
    insights.append("You are Overspending!")
    #suggest adjustments
    reduction_needed=abs(savings)
    insights.append(f"Try Reducing expenses by ₹{int(reduction_needed)}")
  if savings > 0.3 * salary:
        insights.append(" Great saving habits!")
  elif savings > 0.1 * salary:
        insights.append(" Decent savings, can improve more")
  
  result={
      "Rent":rent,
      "Food":food,
      "Transport":transport,
      "Entertainment":entertainment,
      "Other":other,
      "Total Expenses":total_expenses,
      "Savings":savings,
      "Insights":insights
  }
  return result

#user input
# prediction=predict_expense(user,model,encoder,scaler)
# final_output=analyze_expense(user,prediction)
# print(final_output)
