import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset =pd.read_csv(r"S:\Nareshit\classes material\5.may\my_classification_practicals\Churn_Modelling.csv")

X =dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')

X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)


y_pred =classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

bias = classifier.score(X_train, y_train)
print(bias)


variance = classifier.score(X_test, y_test)
print(variance)