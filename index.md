## Project 1

The goal of project 1 is to evaluate the performance of different regressors when predicting the price of a property using the number of rooms. The regressors I examined were: linear regressions, kernel weighted regressions, neural networks, XGBoost, and support vector machines. I used K-fold cross validation to examine each model's mean absolute error. The model that yields the lowest mean absolute error when predicting the price would be the most accurate. 

### Start

Below, I imported relevant packages and pulled in the data set. I also set the parameter value k equal to 10 for my K-fold cross validation. K-fold cross validation shuffles the data set, splits the data sets into k groups (10 in my case), evaluates each group as its own individual test set and using the remaining groups as a training set to fit the model, and retains an evaluation score for each group. Ultimately, the accuracy of a model is summarized by the sample of the model evaluation scores collected for each group. 
```python
import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True, random_state=2021)
df = pd.read_csv('/DATA 410/Boston Housing Prices.csv')
```
Here, I performed preprocessing and split the data into training and test sets with a test set size of .3 or 30%, and a random state of 2021.
```python 
from sklearn.model_selection import train_test_split
X = np.array(df['rooms']).reshape(-1,1)
y = np.array(df['cmedv']).reshape(-1,1)
dat = np.concatenate([X,y.reshape(-1,1)], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)
dat_train = np.concatenate([X_train,y_train.reshape(-1,1)], axis=1)

y_train = y_train.reshape(len(y_train),)
y_test = y_test.reshape(len(y_test),)
```
### Linear Regression

Linear regressions model the linear relationship between a dependent variable and independent variable(s). This method calculates the best-fitting line for the observed data by minimizing the sum of the squares of the vertical deviations from each data point to the line (also known as the Sum of Squared Residuals or the Sum of Squared Errors). Below I fit the linear regression and found the mean absolute error of the model's predictions using K-fold cross validation.
```python 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error 
lm = LinearRegression() 
mae_lm = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  lm.fit(X_train.reshape(-1,1),y_train)
  yhat_lm = lm.predict(X_test.reshape(-1,1))
  mae_lm.append(mean_absolute_error(y_test, yhat_lm))
print("Validated MAE Linear Regression = ${:,.2f}".format(1000*np.mean(mae_lm)))
```
Validated MAE Linear Regression = $4,433.17 

### Kernel Weighted Regression (Loess)

Below I have the kernels that I will be using for my Kernel Weighted Regressions: Tricubic, Epanechnikov, Quartic, Cosine, Triweight, Gaussian. These kernels use their respective functions to determine the weights of our data points for our locally weighted regression. Kernel weighted regressions work well for data that does not show linear qualities. 
```python 
def tricubic(x):
  return np.where(np.abs(x)>1,0,70/81*(1-np.abs(x)**3)**3)

def Epanechnikov(x):
  return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2)) 

def Quartic(x):
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2)

def Cosine(x): 
  return np.where(np.abs(x)>1,0,(np.pi/4)*np.cos((np.pi/2)*np.radians(np.abs(x))))

def Triweight(x):
  return np.where(np.abs(x)>1,0,35/32*(1-abs(x)**2)**3)

def Gaussian(x):
  return np.where(np.abs(x)>1,0,np.exp(-1/2*x**2))
```
The LOESS model performs linear regressions on subsets of data that is weighted by a kernel function. Below, I intialized the LOESS regression model. 
```python
def lowess_kern(x, y, kern, tau):
    n = len(x)
    yest = np.zeros(n)  
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        theta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 

    return yest
def model_lowess(dat_train,dat_test,kern,tau):
  dat_train = dat_train[np.argsort(dat_train[:, 0])]
  dat_test = dat_test[np.argsort(dat_test[:, 0])]
  Yhat_lowess = lowess_kern(dat_train[:,0],dat_train[:,1],kern,tau)
  datl = np.concatenate([dat_train[:,0].reshape(-1,1),Yhat_lowess.reshape(-1,1)], axis=1)
  f = interp1d(datl[:,0], datl[:,1],fill_value='extrapolate')
  return f(dat_test[:,0])
```

Below, I fit various LOESS models, each using a different kernel function to determine the weights of data points. I then used K-fold validation to find the mean absolute error of the kernel weighted regressions' predicitons.  

```python
mae_lke = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lke = model_lowess(dat[idxtrain,:],dat[idxtest,:],Epanechnikov,0.45)
  mae_lke.append(mean_absolute_error(y_test, yhat_lke))
print("Validated MAE Local Epanechnikov Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lke)))
```
Validated MAE Local Epanechnikov Kernel Regression = $4,113.99
```python 
mae_lktc = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lktc = model_lowess(dat[idxtrain,:],dat[idxtest,:],tricubic,0.45)
  mae_lktc.append(mean_absolute_error(y_test, yhat_lktc))
print("Validated MAE Local Tricubic Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lktc)))
```
Validated MAE Local Tricubic Kernel Regression = $4,110.37
```python 
mae_lkq = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lkq = model_lowess(dat[idxtrain,:],dat[idxtest,:],Quartic,0.45)
  mae_lkq.append(mean_absolute_error(y_test, yhat_lkq))
print("Validated MAE Local Quartic Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lkq)))
```
Validated MAE Local Quartic Kernel Regression = $4,107.47
```python 
mae_lkc = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lkc = model_lowess(dat[idxtrain,:],dat[idxtest,:],Cosine,0.45)
  mae_lkc.append(mean_absolute_error(y_test, yhat_lkc))
print("Validated MAE Local Cosine Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lkc)))
```
Validated MAE Local Cosine Kernel Regression = $4,125.13
```python 
mae_lktw = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lktw = model_lowess(dat[idxtrain,:],dat[idxtest,:],Triweight,0.45)
  mae_lktw.append(mean_absolute_error(y_test, yhat_lktw))
print("Validated MAE Local Triweight Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lktw)))
```
Validated MAE Local Triweight Kernel Regression = $4,110.78
```python 
mae_lkg = []

for idxtrain, idxtest in kf.split(dat):
  dat_test = dat[idxtest,:]
  y_test = dat_test[np.argsort(dat_test[:, 0]),1]
  yhat_lkg = model_lowess(dat[idxtrain,:],dat[idxtest,:],Gaussian,0.45)
  mae_lkg.append(mean_absolute_error(y_test, yhat_lkg))
print("Validated MAE Local Gaussian Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_lkg)))
```
Validated MAE Local Gaussian Kernel Regression = $4,118.57

The weighted regression using the Quartic kernel performed the best and yielded the lowest mean absolute error out of all the kernel weigthed regressions tested. 

### Neural Net

Neural Networks are models that look to recognize underlying relationships in a data set through a process that is similar to the way that the human brain works. Neural networks use activation functions to transform inputs. In a neural network, a neuron is a mathematical function that collects and classifies information according to a specific structure or architecture. The neural network ultimately goes through a learning process in which it fine tunes the connection strengths and relationships between neurons in the network to optimize the neural networks performance in solving a particular problem, which in our case is predicting the price. Below I used K-fold validation to find the absolute mean error for the predictions of the neural network model below.  

```python 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
```
```python 
model = Sequential()
model.add(Dense(128, activation="relu", input_dim=1))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(loss='mean_absolute_error', optimizer=Adam(lr=1e-3, decay=1e-3 / 200))
mae_nn = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
  model.fit(X_train.reshape(-1,1),y_train,validation_split=0.3, epochs=1000, batch_size=100, verbose=0, callbacks=[es])
  yhat_nn = model.predict(X_test.reshape(-1,1))
  mae_nn.append(mean_absolute_error(y_test, yhat_nn))
print("Validated MAE Neural Network Regression = ${:,.2f}".format(1000*np.mean(mae_nn)))
```
Validated MAE Neural Network Regression = $4,281.78

### XGBoost

Boosting refers to a family of algorithms that look to turn weak learners into strong learners. In boosting, the individual models are built sequentially by putting more weight on instances where there are wrong predictions and high magnitudes of errors. The model will focus during learning on instances which are hard to predict correctly, so that the model in a sense learns from past mistakes. Extreme gradient boost is a decision-tree based algorithm that uses advanced gradient boosting and regularization to prevent overfitting. I used K-fold validation to find the mean absolute error for the XGBoost model's predictions.

```python 
import xgboost as xgb
model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
mae_xgb = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  model_xgb.fit(X_train.reshape(-1,1),y_train)
  yhat_xgb = model_xgb.predict(X_test.reshape(-1,1))
  mae_xgb.append(mean_absolute_error(y_test, yhat_xgb))
print("Validated MAE XGBoost Regression = ${:,.2f}".format(1000*np.mean(mae_xgb)))
```
Validated MAE XGBoost Regression = $4,136.63

### Support Vector Machine 

The support vector machine first uses the specific kernel to transform the data so that it can be seperated. From there, the algorithm looks to find a hyperplane that seperates the two classes, and more specifically, using support vectors (data points that are closer to the hyperplane and influence the position and orientation of the hyperplane) seeks to find the optimal hyperplane that has the highest margin, or distance between the two classes. Below are the two SVR models that I used that utilized different kernels (Linear and Radial Basis Function). I was originally planing to do the polynomial kernel as well but the run time took to long so I did not end up using it. I used K-fold validation to find the mean absolute error for the SVR models' predictions. 

```python 
from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
```
```python 
model = svr_lin
mae_svr = []
for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  model.fit(X_train.reshape(-1,1),y_train)
  yhat_svr = model.predict(X_test.reshape(-1,1))
  mae_svr.append(mean_absolute_error(y_test, yhat_svr))
print("Validated MAE Support Vector Regression = ${:,.2f}".format(1000*np.mean(mae_svr)))
```
Validated MAE Support Vector Regression = $4,432.00
```
model = svr_rbf
mae_svr = []
for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  model.fit(X_train.reshape(-1,1),y_train)
  yhat_svr = model.predict(X_test.reshape(-1,1))
  mae_svr.append(mean_absolute_error(y_test, yhat_svr))
print("Validated MAE Support Vector Regression = ${:,.2f}".format(1000*np.mean(mae_svr)))
```
Validated MAE Support Vector Regression = $4,130.50

### Conclusion 

After evaluating the different regressors and the mean absolute error of their predicitons, the most accurate model at predicting price using the number of rooms was the locally weighted regression that used the quartic kernel. After k-fold validation, the quartic kernel weighted regression's mean absolute error of $4,107.47 was the lowest out of all the models tested. The quartic kernel weighted regression not only performed better than the other kernel weighted regressions, but outperformed all the other models tested. 

Conversely, the worst performing model tested was the baseline linear regression, which yielded the highest mean absolute error of $4,433.17. Additionally, The SVM that used a linear kernel also performed poorly with a similar mean absolute error of $4,432.00.

