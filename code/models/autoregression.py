import numpy as np
from numpy.linalg import solve
import statistics


# Ordinary Least Squares
class LeastSquares:
    def fit(self, X, y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

    def get_weights(self):
        return self.w


class LeastSquaresRegularized(LeastSquares):
    def __init__(self, lda):
        self.lda = lda

    def fit(self, X, y):
        _n, d = X.shape
        self.w = solve(X.T@X + self.lda*np.identity(d), X.T@y)


class WeightedLeastSquares(LeastSquares):
    # 1) Compute the gradient of f(w)
    # 2) Set equal to 0
    # 3) Call solve to get self.w
    # 4) Set self.w
    def fit(self, X, y, z):
        # Construct diagonal matrix of the weights
        V = np.diag(z)

        # We have that gradient of f(w) = 2X^TVXw - 2X^TVy
        # Setting equal to 0 we get system X^TVXw = X^TVy
        # Then, apply (X^TVX)^-1 to both sides and get
        # w = (X^TVX)^-1(X^TVy)
        self.w = (np.linalg.inv((X.T@V@X))@(X.T@V@y))

class TimeSeriesRegression():

    def __init__(self, lookback, lda=1):
        self.lookback = lookback
        self.first_prediction = True
        self.last_prediction = 0
        self._internal_model = LeastSquaresRegularized(lda)

    def create(self, df, use_variance=False, use_derivative=False):
        # One of these has to be false
        assert(not use_variance or not use_derivative)

        self.df = df
        self.use_variance = use_variance
        self.use_derivative = use_derivative

        data = df.tolist()
        length_df = len(data)
        

        # Create a matrix of zeros with length_df - lookback rows,
        # and the (number of dataframes * lookback) + 1 (for ones) columns
        num_rows = length_df - self.lookback
        num_cols = self.lookback + 1
        X = np.zeros((num_rows, num_cols))
        

        # Now, we loop over the data and put it into the matrix X
        for k in range(num_rows):
            this_row = [1]
            for i in range(num_cols - 1):
                # assert(int(i/self.lookback) == 0)
                this_row.append(data[k + i])
            X[k] = np.array(this_row)
        
        # Special case where we add on average rate of change or variance to end of each example
        if (use_variance or use_derivative):
            X = np.append(X, np.zeros((num_rows, 1)), axis=1)
            for k in range(num_rows):
                if (self.use_variance):
                    this_var = statistics.variance(X[k, 1:num_cols])
                    X[k, num_cols] = this_var
                elif (self.use_derivative):
                    dx = np.mean(np.diff(X[k, 1:-1]))
                    if dx > 0:
                        this_deriv = 1/dx
                    else:
                        this_deriv = 0

                    X[k, num_cols] = this_deriv

        self.last_X = X[num_rows - 1]
        self.length_df = length_df

        # y is data[k:]
        y = np.array(data[self.lookback:])

        # print(X)
        # print(y)

        self._internal_model.fit(X, y)

    def predict(self, X):
        return self._internal_model.predict(X)

    def predictnext(self):
        # Generate the next example
        # by shifting the last row of X
        last_X = self.last_X
        new_X = np.array([1])
        if (self.use_variance or self.use_derivative):
            new_X = np.append(new_X, last_X[2:-1])
        else:
            new_X = np.append(new_X, last_X[2:])

        if (self.first_prediction):
            new_X = np.append(new_X, np.array(
                self.df.tolist()[self.length_df - 1]))
            self.first_prediction = False
        else:
            new_X = np.append(new_X, self.last_prediction)

        # Special cases where we add on variance or average rate of change
        if (self.use_variance):
            var = statistics.variance(new_X[1:])
            new_X = np.append(new_X, np.array([var]))
        elif (self.use_derivative):
            this_deriv = 1/np.mean(np.diff(new_X[1:]))
            new_X = np.append(new_X, np.array([this_deriv]))

        self.last_prediction = self.predict(np.array(new_X))
        self.last_X = new_X
        return self.last_prediction