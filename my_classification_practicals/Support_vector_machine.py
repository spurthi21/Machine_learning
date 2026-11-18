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

from sklearn.svm import SVC
classifier =SVC()
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

from sklearn.metrics import roc_auc_score, roc_curve


# Create SVC model
classifier = SVC(probability=True)

# Train model
classifier.fit(X_train, y_train)

# Predict probability
y_pred_prob = classifier.predict_proba(X_test)[:, 1]

# AUC Score
auc_score = roc_auc_score(y_test, y_pred_prob)

print("AUC Score:", auc_score)

# ROC Curve values
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC Curve
plt.figure(figsize=(8,6))

plt.plot(fpr, tpr,
         label=f'SVC (AUC = {auc_score:.2f})')

plt.plot([0,1], [0,1], 'k--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVC')
plt.legend(loc='lower right')
plt.grid()

plt.show()