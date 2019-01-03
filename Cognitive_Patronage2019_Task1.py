import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('salary.csv', na_values = ['',' '])
X = df['workedYears'].values.reshape(-1,1)
y = df['salaryBrutto'].values.reshape(-1,1)
df.fillna(0, inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
X_train = X[:47]
X_test = X[47:]
y_train = y[:47]
y_test = y[47:59]

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.predict(X_test)

predicted_salary = lm.predict(X_test)
listsalary = predicted_salary.round(2).tolist()
years_to_predict = X_test.tolist()

for i in range(len(listsalary)):
    print("For %s years of experience, predicted salary will be %s." % (*years_to_predict[i], *listsalary[i]))

plt.xlabel('Worked years')
plt.ylabel('Predicted salary brutto')
plt.plot(X_test, predicted_salary)
plt.show()