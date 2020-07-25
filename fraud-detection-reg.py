# Credit Card Fraud Detection in Python using
# Linear & Logistic Regression

#=========================================================================
# Author: Marjan Khamesian
# Date: July 2020
#=========================================================================

import numpy as np
import pandas as pd

# Load the dataset
data=pd.read_csv('../input/creditcardfraud/creditcard.csv')  
print(data.head())

# Data information
print(data.info())

# Shape of the data
print(data.shape)
print(data.describe())

# Checking missing values
print('Missing Values', data.isnull().values.any())
# ------------------------------------------------------------------------
# Feature 'Class':
# A fraud transaction when its value equals 1.
# A valid transaction when its value is 0.

# Fraud & non-fraud 
obs = data['Class'].value_counts()

# Ratio of fraud & non-fraud  
print(obs/len(data))
# ------------------------------------------------------------------------
# Comparison between fraud and non-fraud cases 
import matplotlib.pyplot as plt 
plt.xlabel('Class')
plt.ylabel('Number of Observations')
obs.plot(kind = 'bar',title = '',rot=0)

# Comparison 
plt.scatter(data.loc[data['Class'] == 0]['V11'], data.loc[data['Class'] == 0]['V12'], label='Class #0', alpha=0.5, linewidth=0.15,c='g')
plt.scatter(data.loc[data['Class'] == 1]['V11'], data.loc[data['Class'] == 1]['V12'], label='Class #1', alpha=0.5, linewidth=0.15,c='r')

plt.show()
# ------------------------------------------------------------------------
# Visualization
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(18,4))

# Distribution of Transaction Time
sns.distplot(data['Time'].values/(60*60), ax=ax[0], color='g')
ax[0].set_title('Distribution of Transaction Time', fontsize=14)
ax[0].set_xlim([min(data['Time'].values/(60*60)), max(data['Time'].values/(60*60))])

# Distribution of Transaction Amount: the price of each transaction
sns.distplot(data['Amount'].values, ax=ax[1], color='r')
ax[1].set_title('Distribution of Transaction Amount', fontsize=14)
ax[1].set_xlim([min(data['Amount'].values), max(data['Amount'].values)])

plt.show()
# ------------------------------------------------------------------------
# Number of fraud cases and non-fraud(Valid) transactions
Fraud = data[data['Class'] == 1]  
Valid = data[data['Class'] == 0]  

# Outlier
Outlier_Fraction = len(Fraud)/float(len(Valid))
print('Outlier:', Outlier_Fraction)
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))

# ------------------------------------------------------------------------
# Fraud transactions details
print('Amount details of fraudulent transaction')
print(Fraud.Amount.describe())

# Valid transactions details
print('details of valid transaction')
print(Valid.Amount.describe())

# ------------------------------------------------------------------------
X = data.drop('Class',axis=1)  # Features 
y = data.Class                 # Class variable

# Splitting the data into training/testing sets
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# ========================================================================
#                           Linear Regression
# ========================================================================
from sklearn import datasets, linear_model

linreg = linear_model.LinearRegression()

# Fitting: Training the model using the training sets
linreg.fit(X_train, y_train)

# Prediction 
y_pred_linreg = lreg.predict(X_test)

# Coefficients
print('Coefficients: \n', linreg.coef_,'\n')

# =======================================================================
#                           Model Evaluation
# =======================================================================
from sklearn.metrics import mean_squared_error, r2_score

# Mean Squared Error
print('Mean squared error: %.4f'
      % mean_squared_error(y_test, y_pred_linreg), '\n')

# Coefficient of Determination
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred_linreg))

# Model Evaluation Metrics 
import sklearn.metrics as metrics

# MAE, MSE, RMSE
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_linreg))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_linreg))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_linreg)))

# =========================================================================
#                          Logistic Regression
# =========================================================================
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

# Fitting 
logreg.fit(X_train, y_train.values.ravel())

# Prediction
y_pred_logreg = logreg.predict(X_test)

# Coefficients
print('Coefficients: \n', logreg.coef_,'\n')

# =========================================================================
#                           Model Evaluation
# =========================================================================
# Mean Squared Error
print('Mean squared error: %.4f'
      % mean_squared_error(y_test, y_pred_logreg),'\n')

# Coefficient of Determination
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred_logreg))

# Model Evaluation Metrics 
import sklearn.metrics as metrics

# MAE, MSE, RMSE
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_logreg))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_logreg))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_logreg)))

# Classification Report and Confusion Matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

classification_report = classification_report(y_test, y_pred_logreg) 
confusion_matrix = confusion_matrix(y_test, y_pred_logreg)

print('Classification Report:', classification_report) 
print('Confusion Matrix', confusion_matrix) 
