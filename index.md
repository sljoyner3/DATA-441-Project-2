# Multidimensional Locally Weighted Linear Regression
## DATA 441 Project #2 - Sam Joyner

For this project, we were tasked with adapting Alex Gramfort's 1-dimension locally weighted linear regression (LOWESS) function to work for multi-dimensional data. With this page I will highlight my multi-dimensional LOWESS function, as well as how it was made to be SciKitLearn compliant which allowed me to easily validate my output mean square error using K-Fold cross validation. Lastly, I will also demonstrate GridSearchCV, a SciKitLearn function that makes it easy to identify the optimal hyperparameter values.

The following imports are used with various elements of the code fragments and methods in this page, and are included if the reader so desires to run the code themselves.
``` Python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import scipy.stats as stats 
from sklearn.model_selection import train_test_split as tts, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error as mse
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from math import ceil
from scipy import linalg
import scipy.stats as stats
# Needed to make SciKitLearn compliant function
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
```

#### Adapting Gramfort's Function to Multiple Dimensions

I began with Gramfort's original function, and changed it as necessary for it to work with increased dimensionality. The main changes were evaluating the distance from each point to every other point using Euclidean distances instead of absolute difference for the distances and weights calculations, and adjusting the computation of the linear regressions to work with matrices, particularly ensuring that the dimensions would be compatible for multiplication and adding L2 regression in case the A matrix is not invertible. I also added clauses for adjusting certain vectors to column vectors where necessary. I also adjusted the calculation of delta to work with multi-dimensional matrices. Lastly, I also added the necessary interpolators at the end to be able to use the model to make predictions. The method itself can be seen below with detailed comments to explain what is happening at each code block:

```Python
def gramfort_lowess_multidimensional(x_train, y_train, x_test, f=2. / 3., iter = 3, 
                                     scale = False, a = 6.0, intercept = True):
  
  # Give the option to scale the x datasets using StandardScaler if desired
  if scale == True:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

  # Based on the length of the x training dataset and the f hyperparameter determine
  # how many points should be in each neighborhood
  n = len(x_train)
  r = int(ceil(f * n))

  # Initialize yest to all zeros and delta to all ones
  yest = np.zeros(n)
  delta = np.ones(n)

  # If x_train or y_train are one dimensional force them to column vectors
  if len(x_train.shape)==1:
    x_train = x_train.reshape(-1,1)
  if len(y_train.shape)==1:
    y_train = y_train.reshape(-1,1)

  # Add a leading column of ones to x_train to add the intercept to the mathmatical calculations
  x_train_stacked = x_train
  if intercept:
    x_train_stacked = np.column_stack([np.ones((len(x_train),1)),x_train])

  # For each point calculate the maximum Euclidean distance away a point can be to be included
  # in the neighborhood using r
  h = [np.sort(np.sqrt(np.sum((x_train-x_train[i])**2,axis=1)))[r] for i in range(len(x_train))]

  # Calculate a weights matrix where the ith row is the distance of every other point from the ith point
  # and then clip the matrix to 0.0 and 1.0
  w = np.clip(np.array([np.sqrt(np.sum((x_train-x_train[i])**2,axis=1)) for i in range(len(x_train))]) / h, 0.0, 1.0)
  # Recalculate w so that large distances get a weight of zero, or close to it, and small/zero distances 
  # get a weight of one, or close to it
  w = (1 - w ** 3) ** 3

  # Iterate for the specified number of times across the entire dataset
  for iteration in range(iter):
    # For each value in the dataset compute the linear regression values
    # to find the coefficients, then update the yest for that value
    for i in range(n):
        # Multiply the diagonal weights matrix by delta for each iteration to give
        # more weight to estimates with lower residuals, delta is a column vector
        # with a value for each weight (diagonal value) in the w matrix, so values
        # with lower residuals get more weight
        W = delta * np.diag(w[i,:])
        b = np.transpose(x_train_stacked).dot(W).dot(y_train)
        A = np.transpose(x_train_stacked).dot(W).dot(x_train_stacked)
        A = A + 0.0001*np.eye(x_train_stacked.shape[1])
        beta = linalg.solve(A, b)
        yest[i] = np.dot(x_train_stacked[i],beta)
    
    # After each iteration calculate the residuals, then find the median, and standardize the data
    # based on the median and hyperparameter a so that values lie in between -1 and 1 based on relative
    # closeness to median, then use a function to make values with absolute values close to 1 have values
    # closer to 0 and values closer to 0 would map closer to 1
    residuals = y_train - yest.reshape(-1,1) # Reshape so that yest is a column vector
    s = np.median(np.abs(residuals))
    delta = np.clip(residuals / (a * s), -1, 1)
    delta = (1 - delta ** 2) ** 2
    # Delta is now a column vector of standardized/clipped/adjusted residuals that will be 
    # used for future iterations

  # For applying model to new data
  # If the data is 1-dimensional
  if x_train.shape[1]==1:
    f = interp1d(x_train.flatten(),yest,fill_value='extrapolate')
  # If the data is more than 1-dimensions
  else:
    f = LinearNDInterpolator(x_train, yest)
  # Based on the interpolator that was used, get the output
  output = f(x_test)
  # If there are NaN values, use the following interpolator to fill those values
  if sum(np.isnan(output))>0:
    g = NearestNDInterpolator(x_train,y_train.ravel()) 
    output[np.isnan(output)] = g(x_test[np.isnan(output)])
  
  # Return the estimates based on the new data
  return output
  ```
  
  Additional changes from Gramfort's original function include adding a parameter for the 'a' value which is part of the delta calculation, a boolean value for whether to scale the data or not, and a boolean for calculating with or without an intercept.
  
  #### Making this function SciKitLearn Compatible
  
  I used small, fixed tests with a subset of the cars data to ensure that my method worked and was error free, but for large scale testing I sought to implement my function as a SciKitLearn compliant function to allow for easy use of the K-Fold cross validation technique. The following code allows me to use SciKitLearn functions with my multidimensional LOWESS code: 
  
  ``` Python
  class Gramfort_LOWESS_Multidimensional:
    def __init__(self, f = 1/4, iter = 3, scale = False, a = 6.0, intercept=True):
        self.f = f
        self.iter = iter
        self.scale = scale
        self.a = a
        self.intercept = intercept
    
    def fit(self, x, y):
        f = self.f
        iter = self.iter
        scale = self.scale
        a = self.a
        intercept = self.intercept
        self.xtrain_ = x
        self.yhat_ = y

    def predict(self, x_new):
        check_is_fitted(self)
        x = self.xtrain_
        y = self.yhat_
        f = self.f
        iter = self.iter
        scale = self.scale
        a = self.a
        intercept = self.intercept
        return gramfort_lowess_multidimensional(x, y, x_new, f=f, iter=iter, scale=scale,
                                                a=a, intercept=intercept)

    def get_params(self, deep=True):
        return {"f": self.f, "iter": self.iter,"scale":self.scale,"a":self.a,"intercept":self.intercept}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
  ```
  In a nutshell, this code contains the necessary methods to instantiate this 'Gramfort_LOWESS_Multidimensional' class, use it to fit data and make predictions with the previous modeling function, and also allows to user to view and change the parameters. This is all done in a way to make it easy to use K-Fold cross validation to evaluate model quality, and create an overall more usable modeling tool out of this multidimensional LOWESS method.
  
  #### Using K-Fold Cross-Validation to Evaluate Model Quality
  
  Using my custom method and the SciKitLearn implementation I used the following code to get a cross validated MSE score for my model tested on the 'cars' dataset. This model uses the engine size, number of cylinders, and car weight to predict the mileage (MPG) of a vehicle. I determined the hyperparameters through smaller, non-validated but quick, tests and then applied them here. I also scaled the values so that large values, like car weight (lbs), did not outweigh engine size and cylinder count. I also left the intercept as true to include this in the calculation.
  
  ``` Python
  cars = pd.read_csv('/content/cars.csv')
x = cars.loc[:,'CYL':'WGT'].values
y = cars['MPG'].values

mse_lwr = []
kf = KFold(n_splits=10,shuffle=True,random_state=1234)
model_lw = Gramfort_LOWESS_Multidimensional(f=1/8,iter=10,scale=True,a=10.0,intercept=True)

for idxtrain, idxtest in kf.split(x):
  xtrain = x[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = x[idxtest]

  model_lw.fit(xtrain,ytrain)
  yhat_lw = model_lw.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))

print('Cars Data Locally Weighted Regression K-Fold validated MSE: '+str(np.mean(mse_lwr)))
  ```
The output MSE for this was 17.3299 miles per gallon squared. This indicates a decently accurate model that could likely reliably predict a car's MPG within 4 or 5 miles of the true value. Another example of this model is below, with the concrete dataset where some of the ingredients and properties of concrete are used to predict its strength:

``` Python
concrete = pd.read_csv('/content/concrete.csv')
x = concrete.loc[:,'cement':'water'].values
y = concrete['strength'].values

mse_lwr = []
kf = KFold(n_splits=4,shuffle=True,random_state=1234)
model_lw = Gramfort_LOWESS_Multidimensional(f=1/4,iter=3,scale=False,a=6.0,intercept=True)

for idxtrain, idxtest in kf.split(x):
  xtrain = x[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = x[idxtest]

  model_lw.fit(xtrain,ytrain)
  yhat_lw = model_lw.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))

print('Concrete Data Locally Weighted Regression K-Fold validated MSE: '+str(np.mean(mse_lwr)))
```
The validated MSE for this data with the parameters above was 145.1976, which is not bad, but seems to have room for improvement. The best way to improve the results are by tuning the hyperparameters, but this can be a daunting task to know how to change the values to improve the model.

#### Gridsearch

To make this easier, there is a SkiLearnFunction called GridSearchCV. This function allows for one to define their data pipeline and a range of values for parameters, and the gridsearch will iterate through and try various parameter combinations to identify the optimal values. A demonstration of this can be seen with the code below for the concrete dataset, where I will test values around the hyperparameters I began with to see if a more optimal set of values exists:

```Python
concrete = pd.read_csv('/content/concrete.csv')
x = concrete.loc[:,'cement':'water'].values
y = concrete['strength'].values

lwr_pipe = Pipeline([('zscores', StandardScaler()),
                     ('lwr', Gramfort_LOWESS_Multidimensional())])

params = [{'lwr__f': [1/i for i in range(2,8,2)],
           'lwr__iter': [1,3,5],
           'lwr__a': [4,6,8]}]

gs_lowess = GridSearchCV(lwr_pipe,
                         param_grid=params,
                         scoring='neg_mean_squared_error',
                         cv=4)
gs_lowess.fit(x, y)
gs_lowess.best_params_
```

The gridsearch has highlighted the following parameter values as ideal: . Plugging these into the otherwise identical K-Fold cross validation as before now yields an improved MSE of , highlighting the usefulness of gridsearch in optimizing hyperparameters of model.

#### Main Takeaways

I was able to modify Gramfort's LOWESS to work with multidimensional datasets by reworking some of the key mathematical and updating portions of the method. I was then able to implement this as a SciKitLearn compliant function that made it easy to test the model with K-Fold cross validations and use GridSearchCV to identify the optimal hyperparameters. I used the cars and concrete datasets to highlight the K-Fold and gridsearch implementations with real multidimensional data.

