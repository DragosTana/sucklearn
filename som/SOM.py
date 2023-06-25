import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from typing import Optional, Tuple


class SOMCluster(BaseEstimator, ClusterMixin):
    """
    Self-Organizing Map clustering algorithm compatible with scikit-learn.
    
    ## Parameters
    n_clusters : int
        Number of clusters.
        
    map_size : tuple of int, default=(10, 10)
        Size of the map.
        
    sigma : float, default=1.0
        Standard deviation of the Gaussian function that controls the decay of learning rate.
        
    learning_rate : float, default=0.5
        Initial learning rate.

    max_iter : int, default=100
        Maximum number of iterations.
    
    tol : float, default=1e-4
        Tolerance for stopping criteria.
        
    random_state : int, default=None
        Random state.
    """
    def __init__(self,
                n_clusters: int = 8,
                map_size: Tuple[int, int] = (10, 10),
                sigma: float = 1.0,
                learning_rate: float = 0.5,
                max_iter: int = 100,
                tol: float = 1e-4,
                random_state: Optional[int] = None ):
        
        self.n_clusters = n_clusters
        self.map_size = map_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.random_state_ = None

    def _initialize_weights(self, X: np.ndarray) -> None:
        """
        Initialize the weights.
        """
        self.random_state_ = check_random_state(self.random_state)
        self.weights_ = self.random_state_.randn(*self.map_size, X.shape[1])

    def _find_bmu(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Find the best matching unit (BMU) for the given input.
        """
        bmu_idx = np.linalg.norm(self.weights_ - x, axis=2).argmin()
        return np.unravel_index(bmu_idx, self.map_size)

    def _update_weights(self, x: np.ndarray, bmu: Tuple[int, int]) -> None:
        """
        Update the weights.
        """
        distances = np.linalg.norm(np.indices(self.map_size).T - bmu, axis=2)
        influence = np.exp(-distances**2 / (2 * self.sigma**2))
        learning_rate = self.learning_rate * np.exp(-self.iteration / self.max_iter)
        delta = learning_rate * influence[:, :, np.newaxis] * (x - self.weights_)
        self.weights_ += delta

    def _calculate_quantization_error(self, X: np.ndarray) -> float:
        """
        Calculate the quantization error.
        """
        bmu_indices = np.array([self._find_bmu(x) for x in X])
        quantization_error = np.mean(np.linalg.norm(X - self.weights_[bmu_indices], axis=1))
        return quantization_error

    def fit(self, X: np.ndarray) -> 'SOMCluster':
        """
        Fit the SOMCluster model according to the given training data.
        """
        self._initialize_weights(X)
        self.iteration = 0
        prev_error = None

        while self.iteration < self.max_iter:
            indices = self.random_state_.permutation(len(X))
            for i in indices:
                x = X[i]
                bmu = self._find_bmu(x)
                self._update_weights(x, bmu)

            quantization_error = self._calculate_quantization_error(X)
            if prev_error is not None and abs(prev_error - quantization_error) < self.tol:
                break

            prev_error = quantization_error
            self.iteration += 1

        self.cluster_centers_ = self.weights_.reshape(-1, X.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        """
        bmu_indices = np.array([self._find_bmu(x) for x in X])
        labels = np.ravel_multi_index(bmu_indices, self.map_size)
        return labels
    
    def get_neuron_coordinates(self) -> np.ndarray:
        """
        Get the coordinates of the neurons.
        """
        return np.indices(self.map_size).reshape(2, -1).T

    def get_neuron_weights(self) -> np.ndarray:
        """
        Get the weights of the neurons.
        """
        return self.weights_.reshape(-1, self.weights_.shape[2])
