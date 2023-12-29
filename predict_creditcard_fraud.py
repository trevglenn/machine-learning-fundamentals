import seaborn
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the data
transactions = pd.read_csv('transactions_modified.csv')
#print(transactions.head())
#print(transactions.info())

# How many fraudulent transactions?
#print(transactions['isFraud'].sum())
# 282 fraudulent transactions

# Summary statistics on amount column
print(transactions['amount'].describe())

# Create isPayment field
def det_isPayment(x):
  if x == 'PAYMENT':
    return 1
  if x == 'DEBIT':
    return 1
  else:
    return 0

vec = np.vectorize(det_isPayment)
isPayment = vec(transactions['type'])
transactions['isPayment'] == isPayment

# Create isMovement field
def det_isMovement(x):
  if x == 'CASH_OUT':
    return 1
  elif x == 'TRANSFER':
    return 1
  else:
    return 0

vec = np.vectorize(det_isMovement)
isMovement = vec(transactions['type'])
transactions['isMovement'] = isMovement
# Create accountDiff field
transactions['accountDiff'] = transactions['oldbalanceOrg'] - transactions['oldbalanceDest']

# Create features and label variables
features = transactions[['amount', 'isPayment', 'isMovement', 'accountDiff']]
label = transactions['isFraud']

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=23)

# Normalize the features variables
sc = StandardScaler()

# Fit the model to the training data
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

regr = LogisticRegression()
regr.fit(x_train_scaled, y_train)
# Score the model on the training data
print(regr.score(x_train_scaled, y_train))

# Score the model on the test data
print(regr.score(x_test_scaled, y_test))

# Print the model coefficients
print(regr.coef_)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
transaction4 = np.array([684321.94, 1.0, 0.0, 45415.33])

# Combine new transactions into a single array
sample_transactions = np.stack([transaction1, transaction2, transaction3, transaction4])

# Normalize the new transactions
sample_transactions = sc.transform(sample_transactions)

# Predict fraud on the new transactions
print(regr.predict(sample_transactions))

# Show probabilities on the new transactions
print(regr.predict_proba(sample_transactions))