
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from kernel_regression.kernel import CustomKernel


class KernelRegression(BaseEstimator, RegressorMixin):
    """
    kernel_type: "gaussian", "uniform", "triangular", "epanechnikov", "cosine"
    bandwidth: float
    
    Kernel regression model compatible with scikit-learn. 
    """

    def __init__(self, kernel_type='gaussian', bandwidth=0.25, reg_type='nadaraya_watson'):
        self.bandwidth = bandwidth
        self.kernel_type = kernel_type
        self.kernel = CustomKernel(kernel_type=kernel_type)
        self.reg_type = reg_type

    def fit(self, X, y):
        """
        Fit the KernelRegression model according to the given training data.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
        self.X_ = np.array(X)
        self.y_ = np.array(y)
        return self

    def predict(self, X):
        """
        Predict regression target for X.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        if self.reg_type == 'nadaraya_watson':
            pred = []
            for x in X:
                tmp = (1 / self.bandwidth) * self.kernel((x - self.X_) / self.bandwidth)
                pred.append(np.dot(tmp.T, self.y_) / np.sum(tmp))
            return pred

        if self.reg_type == 'Priestley_Chao':
            pred = []
            for x in X:
                tmp = 0
                for i in range(1, len(self.X_)):
                    tmp += (self.X_[i] - self.X_[i-1]) * self.kernel((x - self.X_[i]) / self.bandwidth) * self.y_[i]
                pred.append((1 / self.bandwidth) * tmp)
            return pred
