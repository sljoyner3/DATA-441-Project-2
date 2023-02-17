# Multidimensional Locally Weighted Linear Regression
## DATA 441 Project #2 - Sam Joyner

For this project, we were tasked with adapting Alex Gramfort's 1-dimension locally weighted linear regression (LOWESS) function to work for multi-dimensional data. With this page I will highlight my multi-dimensional LOWESS function, as well as how it was made to be SciKitLearn compliant which allowed me to easily validate my output mean square error using K-Fold cross validation. Lastly, I will also demonstrate GridSearchCV, a SciKitLearn function that makes it easy to identify the optimal hyperparameter values.

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
