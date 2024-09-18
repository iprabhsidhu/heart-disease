import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns

#DATASET
disease_df = pd.read_csv("framingham.csv")
disease_df.drop(['education'], inplace=True, axis=1)
disease_df.rename(columns={'male':'Sex_male'}, inplace=True)

disease_df.dropna(axis=0, inplace=True)
print(disease_df.head(), disease_df.shape)
print(disease_df.TenYearCHD.value_counts())

X = np.asarray(disease_df[['age','Sex_male','cigsPerDay','totChol','sysBP','glucose']])
Y = np.asarray(disease_df['TenYearCHD'])

#Normalising DataSET
X = preprocessing.StandardScaler().fit(X).transform(X)

#Train-Test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=4)
print(f'Train set : {X_train.shape},{Y_train.shape}')
print(f'Test set : {X_test.shape},{Y_test.shape}') 

#counting no. of patients affected with CHD
plt.figure(figsize=(7,5))
sns.countplot(x='TenYearCHD',data=disease_df,palette='BuGn_r')
plt.show()

# Plotting TenYearCHD
disease_df['TenYearCHD'].plot()
plt.show()

logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
y_pred = logreg.predict(X_test)

print(f"Accuracy of the model is = {accuracy_score(Y_test, y_pred)}")

cm = confusion_matrix(Y_test, y_pred)
conf_matrix = pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize=(8,5))
sns.heatmap(conf_matrix, annot=True, fmt='d',cmap='Greens')

plt.show()
print('The details for confusion matrix is =')
print(classification_report(Y_test, y_pred))
