import pandas as pd

MYdata=pd.read_csv("C:\\Users\\hp\\DATA_SCIENCE_NOTES\\FLASK\\Income_Expense_Data.csv")

print(MYdata)
MYdata["Income"].fillna(MYdata["Income"].median(),inplace=True)
print(MYdata)
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
scaled_data=scaler.fit_transform(MYdata)
print(scaled_data)

MYdata_scaled = pd.DataFrame(scaled_data)
MYdata_scaled.columns = ["Age","Income","Expense"]
print(MYdata_scaled)


features = ["Income","Age"]
response = ["Expense"]

X=MYdata_scaled[features]
y=MYdata_scaled[response]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.25)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

import pickle
pickle.dump(model, open('model1.pkl','wb'))







