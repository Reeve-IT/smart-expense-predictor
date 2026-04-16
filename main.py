import pandas as pd

exp_df=pd.read_csv("data/indian_consumer_spending_dataset.csv")

exp_df.columns=exp_df.columns.str.strip().str.lower()

from numpy import test
from numpy._core import numeric
from pandas.core.arrays import categorical
#import required libries
from sklearn .preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#define features (X) and (y)
x=exp_df[['user_type','city','tier','salary','family_size','lifestyle']]
y=exp_df[['rent','food','transport','entertainment','other_exp']]

#identify categorical and numerical columns
categorical_cols=['user_type','city','tier','lifestyle']
numerical_cols=['salary','family_size']

 #apply One-Hot Encoding to categorical data
#encoder = OneHotEncoder(sparse=False, drop='first')
encoder=OneHotEncoder(sparse_output=False,drop='first',handle_unknown='ignore')
enncoded_data=encoder.fit_transform(x[categorical_cols])

 #convert encoded data into into DataFrame
encoded_df=pd.DataFrame(enncoded_data,columns=encoder.get_feature_names_out(categorical_cols))

 #combine encoded categorical + numerical data
x_final=pd.concat([encoded_df, x[numerical_cols].reset_index(drop=True)],axis=1)

#feature Scaling (important for ML models)
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x_final)

#train test split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score

#initialize base model
base_model=RandomForestRegressor(n_estimators=100,random_state=42)

#wrap it for multi-output
model=MultiOutputRegressor(base_model)

#train the model
model.fit(x_train,y_train)

#make predictions
y_pred=model.predict(x_test)

#evaluate model
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

# print("MAE:",mae)
# print("R2 Score:",r2)

#convert predictions into df
y_pred_df=pd.DataFrame(y_pred,columns=y.columns)
#reset index for proper alignment
y_test_reset=y_test.reset_index(drop=True)
#combine actual and predicted
comparsion=pd.concat([y_test_reset,y_pred_df],axis=1)
#rename columns for clarity
comparsion.columns=['Actual_Rent','Actual_Food','Actual_Transport','Actual_Entertainment','Actual_Other',
                    'Pred_Rent','Pred_Food','Pred_Transport','Pred_Entertainment','Pred_Other']
#print(comparsion.head())
diffrence=y_test_reset-y_pred_df
#print(diffrence.head())

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
      "Other": prediction[0][4]
  }
  return output

#user input
user={
    'user_type':'Working Professional',
    'city':'Mumbai',
    'tier':'Tier 1-Elite',
    'salary':50000,
    'family_size':5,
    'lifestyle':'Medium'
}
result=predict_expense(user,model,encoder,scaler)
#print(result)

#adding suggestions
def analyze_expense(user_input,prediction):

  salary=user_input['salary']

  rent=prediction['Rent']
  food=prediction['Food']
  transport=prediction['Transport']
  entertainment=prediction['Entertainment']
  other=prediction['Other']

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
prediction=predict_expense(user,model,encoder,scaler)
final_output=analyze_expense(user,prediction)
print(final_output)
