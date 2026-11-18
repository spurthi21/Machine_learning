import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv(r"S:\nareshcode\machinelearning\slr_regression\Salary_Data.csv")

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

comparison = pd.DataFrame({'Actual':y_test,'Prediction':y_pred})
print(comparison)

plt.scatter(x_test, y_test, color = 'Red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary of employee based on experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

y_12yr_exp = m_slope*12+c_intercept
print(y_12yr_exp)


y_20yr_exp = m_slope*20+c_intercept
print(y_20yr_exp)

bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)

# implement statstics in ml regression model

dataset.mean()

dataset['Salary'].mean()
dataset['YearsExperience'].mean()

dataset.median()

dataset['Salary'].median()
dataset['YearsExperience'].median()

#dataset.mode()


dataset.var() # variance
dataset['Salary'].var()
dataset['YearsExperience'].var()

dataset.std() # standard deviation
dataset['Salary'].std()
dataset['YearsExperience'].std()


from scipy.stats import variation
variation(dataset.values)

variation(dataset['Salary'])

dataset.corr() # give corelation od entire dataframe
dataset['Salary'].corr(dataset['YearsExperience'])

# skewness
dataset.skew()
dataset['Salary'].skew()
dataset['YearsExperience'].skew()

# standard error
dataset.sem()
dataset['Salary'].sem()
dataset['YearsExperience'].sem()

# z-score
import scipy.stats as stats
dataset.apply(stats.zscore)

stats.zscore(dataset['Salary'])
stats.zscore(dataset['YearsExperience'])

# ANOVA

y_mean =np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

mean_total =np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

r_square =1-(SSR/SST)
r_square

print(r_square)
print(bias)
print(variance)

import pickle
filename = 'linear_regression_model.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print("model has been pickled and saved as linear_regression_model.pkl")

  



