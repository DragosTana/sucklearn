#Autor: Dragos Tanasa
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import neighbors
from sklearn import datasets
from sklearn import metrics
from scipy import optimize
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import time

import parallelSuperLearner as psl

class SuperLearner(BaseEstimator, RegressorMixin, ClassifierMixin):
    
    """
    Super Learner algorithm for regression and classification tasks.
    
    ## Parameters:
    
    base_estimators: dict
        dictionary of base estimators
        
    meta_learner: estimator, default = None
        meta learner to combine the base estimators' predictions
        
    task: {'regression', 'classification'}, default = 'regression'
        task to perform
        
    threshold: float, default = 0.01
        threshold for the meta learner's coefficients
        
    verbose: bool, default = False
        if True, prints the correlation matrix and scatter matrix of the base estimators' predictions
        
    ## Attributes:

    """
    
    def __init__(self, base_estimators, meta_learner = None, task = 'regression', threshold = 0.01, verbose = False):
        self.base_estimators = base_estimators.values()
        self.meta_learner = meta_learner
        self.threshold = threshold
        self.weights = None
        self.verbose = verbose
        self.task = task
        self.meta_predictions = None
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        
        meta_predictions = np.zeros((X.shape[0], len(self.base_estimators)), dtype=np.float64)
        kf = KFold(n_splits=10)        

        for i, (tran_idx, val_idx) in enumerate(kf.split(X)):
            
            X_train, X_val = X[tran_idx], X[val_idx]
            y_train, y_val = y[tran_idx], y[val_idx]
            for j, estimator in enumerate(self.base_estimators):
                estimator.fit(X_train, y_train)
                meta_predictions[val_idx, j] = estimator.predict(X_val)

        self.meta_predictions = meta_predictions
        
        if self.verbose:
            df = pd.DataFrame(np.hstack((meta_predictions, y.reshape(-1,1))))
            last_column_index = df.shape[1] - 1
            df.rename(columns={last_column_index: 'y'}, inplace=True)
            names = {i : estimator.__class__.__name__ for i, estimator in enumerate(self.base_estimators)}
            df.rename(columns=names, inplace=True)
            print(df.head())
        
            scatter_matrix(df, alpha = 0.2,  figsize = (6, 6), diagonal = 'kde')
            plt.show(block=False)
            print(" ")
        
        if self.task == 'regression':
            self.calculate_weights_regression(meta_predictions, X, y)
        elif self.task == 'classification':
            self.calculate_weights_classification(meta_predictions, X, y)
        
        return self
    
    def calculate_weights_regression(self, meta_predictions, X, y):
        
        if self.meta_learner is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(meta_predictions)
            y_scaled = scaler.fit_transform(y.reshape(-1,1)).flatten()
            result = optimize.nnls(X_scaled, y_scaled)
            result = result[0]
            result = result / np.sum(result)
            result[result < self.threshold] = 0
            result = result / np.sum(result)
            self.weights = result
        else :
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(meta_predictions)
            y_scaled = scaler.fit_transform(y.reshape(-1,1)).flatten()
            self.meta_learner.fit(X_scaled, y_scaled)
            result = self.meta_learner.coef_
            result = result / np.sum(result)
            result[result < self.threshold] = 0
            result = result / np.sum(result)
            self.weights = result
        
        for estimator in self.base_estimators:
            estimator.fit(X, y)
            
        return self
    
    def calculate_weights_classification(self, meta_predictions, X, y):

        accuracies = []
        for i in range(meta_predictions.shape[1]):
            y_pred = meta_predictions[:, i]
            accuracy = metrics.accuracy_score(y, y_pred)
            accuracies.append(accuracy)

        accuracies = np.array(accuracies)
        accuracies_normalized = accuracies / np.sum(accuracies)
        accuracies_normalized[accuracies_normalized < self.threshold] = 0
        self.weights = accuracies_normalized / np.sum(accuracies_normalized)
        
        for estimator in self.base_estimators:
            estimator.fit(X, y)
            
        return self
    
    def predict(self, X):
        check_is_fitted(self, 'meta_learner')
        X = check_array(X)
        
        base_predictions = np.zeros((X.shape[0], len(self.base_estimators)), dtype=np.float64)
        for i, estimator in enumerate(self.base_estimators):
            base_predictions[:, i] = estimator.predict(X)
            
        return np.dot(base_predictions, self.weights)
    
def matrix_distance(matrix1, matrix2):
    array1 = np.array(matrix1)
    array2 = np.array(matrix2)
    difference = array1 - array2
    squared_difference = difference**2
    sum_squared_difference = np.sum(squared_difference)
    distance = np.sqrt(sum_squared_difference)
    
    return distance
        
def main():

    #X, y = datasets.make_friedman1(5000)
    #X, y = datasets.make_friedman2(5000)
    X, y, coef = datasets.make_regression(n_samples=0000, n_features=10, n_informative=5, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=40, shuffle=True, coef=True, random_state=12)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)).flatten()
    
    library1 = {
        "ols": linear_model.LinearRegression(),
        "elastic_0.01": linear_model.ElasticNet(alpha=0.01),
        "elastic_0.1": linear_model.ElasticNet(alpha=0.1),
        "elastic_1.0": linear_model.ElasticNet(alpha=1.0),
        "elastic_10.0": linear_model.ElasticNet(alpha=10.0),
        "ridge_0.01": linear_model.Ridge(alpha=0.01),
        "ridge_0.1": linear_model.Ridge(alpha=0.1),
        "ridge_1.0": linear_model.Ridge(alpha=1.0),
        "ridge_10.0": linear_model.Ridge(alpha=10.0),
        "lasso_0.01": linear_model.Lasso(alpha=0.01),
        "lasso_0.1": linear_model.Lasso(alpha=0.1),
        "lasso_1.0": linear_model.Lasso(alpha=1.0),
        "lasso_10.0": linear_model.Lasso(alpha=10.0),
        "knn_5": neighbors.KNeighborsRegressor(n_neighbors=5),
        "knn_10": neighbors.KNeighborsRegressor(n_neighbors=10),
        "knn_15": neighbors.KNeighborsRegressor(n_neighbors=15),
        "knn_20": neighbors.KNeighborsRegressor(n_neighbors=20),
    }
    
    library2 = {
        "ols": linear_model.LinearRegression(),
        "ridge": linear_model.RidgeCV(alphas=np.arange(0.01, 10.0, 0.01)),
        "lasse" : linear_model.LassoCV(alphas=np.arange(0.01, 10.0, 0.01), positive=True),
        "elastic" :  linear_model.ElasticNetCV(alphas=np.arange(0.01, 10.0, 0.01), positive=True),
        "knn_5": neighbors.KNeighborsRegressor(n_neighbors=5),
        "knn_10": neighbors.KNeighborsRegressor(n_neighbors=10),
        "knn_15": neighbors.KNeighborsRegressor(n_neighbors=15),
        "knn_20": neighbors.KNeighborsRegressor(n_neighbors=20),
    }
        
    library3 = {
        "ols": linear_model.LinearRegression(),
        "elastic_0.1": linear_model.ElasticNet(alpha=0.01),
        "ridge_0.1": linear_model.Ridge(alpha=0.01),
        "lasso_0.1": linear_model.Lasso(alpha=0.01),
        "knn_5": neighbors.KNeighborsRegressor(n_neighbors=5),
        "knn_10": neighbors.KNeighborsRegressor(n_neighbors=10),
    }
        
    
    parallel_sl = psl.SuperLearner(library1)
    sl = SuperLearner(library1)
    start_time = time.time()
    parallel_sl.fit(X_train, y_train)
    end_time = time.time()
    print("Parallel Super Learner took: ", end_time - start_time)
    
    start_time = time.time()
    sl.fit(X_train, y_train)
    end_time = time.time()
    print("Super Learner took: ", end_time - start_time)
    
    if (parallel_sl.meta_predictions == sl.meta_predictions).all():
        print("Meta predictions are equal")
    else:
        #If the meta predictions are not equal, print the distance between the two matrices. 
        #The distance should be very small, but not necessarily zero due to floating point errors
        print("Meta predictions are not equal")
        print("Distance between meta predictions: ", matrix_distance(parallel_sl.meta_predictions, sl.meta_predictions))
        
if __name__ == "__main__":
    main()