from sklearn import linear_model
import numpy as np

X = np.array([200,1300,1500,400,150,1100,900,1500,1450,1100,100,1100,800,300,600])
y = np.array([1,3,4,2,1,3,3,5,4,3,2,4,3,2,2])

X = X.reshape((X.shape[0],1))
y = y.reshape((y.shape[0],1))
reg = linear_model.LinearRegression()
reg.fit(X,y)
print(reg.coef_)
X_test = np.array([750,1000])
X_test = X_test.reshape((X_test.shape[0],1))
print(reg.predict(X_test))