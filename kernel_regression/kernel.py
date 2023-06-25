import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.gaussian_process.kernels import Kernel


class CustomKernel(Kernel):
    """
    Custtom kernel class 
    
    ## Parameters
    kernel_type: str
        Type of kernel to use. Options are 'gaussian', 'uniform', 'triangular', 'epanechnikov', 'cosine'
        
    """
    def __init__(self, kernel_type='epanechnikov', **kwargs):
        self.kernel_type = kernel_type
        self.kwargs = kwargs

    def __call__(self, X, Y=None, eval_gradient=False):
        if self.kernel_type == 'gaussian':
            return np.exp(-X**2 / 2) / np.sqrt(2 * np.pi)
        elif self.kernel_type == 'uniform':
            return np.where(np.abs(X) <= 1, 1/2, 0)
        elif self.kernel_type == 'triangular':
            return np.where(np.abs(X) <= 1, 1 - np.abs(X), 0)
        elif self.kernel_type == 'epanechnikov':
            return np.where(np.abs(X) <= 1, 3/4 * (1 - X**2), 0)
        elif self.kernel_type == 'cosine':
            return np.where(np.abs(X) <= 1, np.pi/4 * np.cos(np.pi * X), 0)
        else:
            raise ValueError('Invalid kernel type')

    def diag(self, X):
        return np.ones(X.shape[0])
