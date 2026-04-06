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
encoder=OneHotEncoder(sparse_output=False,drop='first')
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

print("MAE:",mae)
print("R2 Score:",r2)