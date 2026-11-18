import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset =pd.read_csv(r"S:\Nareshit\classes material\5.may\my_classification_practicals\dataset_for_all_classification.csv")

X =dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

'''from sklearn.preprocessing import Normalizer
nc = Normalizer()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)'''

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
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

# future prediction 

dataset1=pd.read_csv(r"S:\Nareshit\classes material\5.may\my_classification_practicals\Future prediction1.csv")
d2=dataset1.copy()
dataset1=dataset1.iloc[:,[2,3]].values

from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
M =sc.fit_transform(dataset1)

y_pred1 =pd.DataFrame()

d2['y_pred1']=classifier.predict(M)

d2.to_csv('final1.csv')

# to get the path of csv import os , os.getcwd() 

# to get ROC and AUC 

from sklearn.metrics import roc_auc_score,roc_curve
y_pred_prob = classifier.predict_proba(X_test)[:,1]

auc_score = roc_auc_score(y_test, y_pred_prob)
auc_score

fpr,trp,thresholds=roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr,trp,label=f'logistic regression (AUC={auc_score:2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()












