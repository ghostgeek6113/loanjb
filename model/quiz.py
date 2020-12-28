import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pickle
import json


pd.set_option('display.max_rows', 50000000)
pd.set_option('display.max_columns', 50000000)
pd.set_option('display.width', 1000000000)

# loan_data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv")
# loan_data.to_csv("loan_train.csv", index=None)

loan_data = pd.read_csv("loan_train.csv")
# print("------------------------------head-------------------------------")
# print(loan_data.head())
# print("---------------------------info--------------------------")
# print(loan_data.info())
# print("-------------------------description---------------------------------")
# print(loan_data.describe())
# print("----------------------------------null---------------------------------")
# print(loan_data.isna().sum())
loan_data['Gender'].fillna(value='Female', inplace=True)
loan_data['Married'].fillna(value='No', inplace=True)
loan_data['Dependents'].fillna(value='1', inplace=True)
loan_data['Self_Employed'].fillna(value='Yes', inplace=True)
loan_data['LoanAmount'].fillna(value=loan_data['LoanAmount'].mean(), inplace=True)
loan_data['Loan_Amount_Term'].fillna(loan_data['Loan_Amount_Term'].median(), inplace=True)
loan_data['Credit_History'].fillna(value=0, inplace=True)

loan_data = loan_data.drop(columns=['Loan_ID', 'Unnamed: 0'])
loan_data = pd.get_dummies(loan_data,
                           columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
loan_data.to_csv('loan_data.csv')
X = loan_data.drop(columns=['Loan_Status'])
Y = loan_data['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101, stratify=Y)
model = LogisticRegression(max_iter=1000)
model = model.fit(X_train, Y_train)
score = f1_score(Y_test, model.predict(X_test))
# print("Logistic regression")
# print("-------------------")
# print("f1 score: ", score)

loan_test = pd.read_csv('loan_test.csv')
target = model.predict(loan_test)
results = pd.DataFrame(target)
results.columns = ["Predictions"]
results.to_csv("prediction.csv")

# print(results)
with open('Loan_model.pickle', 'wb') as f:
    pickle.dump(model, f)

columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))
