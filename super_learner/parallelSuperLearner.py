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
from sklearn import ensemble
from scipy import optimize
from joblib import Parallel, delayed
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import time

class SuperLearner(BaseEstimator, RegressorMixin, ClassifierMixin):
    """
    Parallel Super Learner algorithm for regression and classification tasks.
    
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
    
    def __init__(self, base_estimators, folds = 10,  meta_learner = None, task = 'regression', threshold = 0.01, verbose = False):
        self.base_estimators = base_estimators.values()
        self.base_estimators_names = base_estimators.keys()
        self.meta_learner = meta_learner
        self.folds = folds
        self.threshold = threshold
        self.weights = None
        self.verbose = verbose
        self.task = task
        self.meta_predictions = None
        
        
    def fit(self, X, y):
        
        X, y = check_X_y(X, y)
        
        meta_predictions = np.zeros((X.shape[0], len(self.base_estimators)), dtype=np.float64)
        kf = KFold(n_splits=self.folds)
        
        def fit_estimator(estimator, X_train, y_train, X_val, val_idx, j):
            estimator.fit(X_train, y_train)
            return estimator.predict(X_val), val_idx, j
        
        results = Parallel(n_jobs=-1)(
            delayed(fit_estimator)(
                estimator, X[tran_idx], y[tran_idx], X[val_idx], val_idx, j
            )
            for i, (tran_idx, val_idx) in enumerate(kf.split(X))
            for j, estimator in enumerate(self.base_estimators)
        )
        
        for result in results:
            meta_predictions[result[1], result[2]] = result[0]
               
        #def train_and_predict(estimator, X_train, y_train, X_val, val_idx):
        #    estimator.fit(X_train, y_train)
        #    return estimator.predict(X_val), val_idx
        #
        #results = Parallel(n_jobs=-1)(
        #    delayed(train_and_predict)(estimator, X[tran_idx], y[tran_idx], X[val_idx], val_idx) 
        #    for tran_idx, val_idx in kf.split(X)
        #    for estimator in self.base_estimators
        #)
        #
        #for meta_pred, val_idx in results:
        #    meta_predictions[val_idx] = meta_pred

        self.meta_predictions = meta_predictions

        if self.verbose:
            df = pd.DataFrame(np.hstack((meta_predictions, y.reshape(-1,1))))
            last_column_index = df.shape[1] - 1
            df.rename(columns={last_column_index: 'y'}, inplace=True)
            #df = pd.DataFrame(meta_predictions)
            names = {i : list(self.base_estimators_names)[i] for i in range(len(list(self.base_estimators_names)))}
            print(names)
            df.rename(columns=names, inplace=True)
            print(df.head(30))
        
            scatter_matrix(df, alpha = 0.2,  figsize = (6, 6), diagonal = 'kde')
            plt.show()
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
            #X_scaled = meta_predictions
            #y_scaled = y
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
            #X_scaled = meta_predictions
            #y_scaled = y
            self.meta_learner.fit(X_scaled, y_scaled)
            result = self.meta_learner.coef_
            result = result / np.sum(result)
            result[result < self.threshold] = 0
            result = result / np.sum(result)
            self.weights = result
        
        def fit_estimator(estimator, X, y):
            estimator.fit(X, y)
            
        Parallel(n_jobs=-1)(
            delayed(fit_estimator)(estimator, X, y)
            for estimator in self.base_estimators)
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
        
def main():
    np.random.seed(0)
    X, y = datasets.make_friedman1(1000, noise=0.5, random_state=0)
    #X, y = datasets.make_regression(n_samples=1000, n_features=30, n_informative=15, n_targets=1, bias=0.0, noise=70, random_state=12)
    #X, y = datasets.load_diabetes(return_X_y=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    y_train = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
    y_test = scaler.fit_transform(y_test.reshape(-1,1)).flatten()
    
    library = {
        "ols": linear_model.LinearRegression(),
        "ridge": linear_model.RidgeCV(alphas=np.arange(0.05, 15, 0.05)),
        "lasso" : linear_model.LassoCV(alphas=np.arange(0.05, 15.0, 0.05)),
        "elastic" :  linear_model.ElasticNetCV(alphas=np.arange(0.05, 15.0, 0.05)),
        "knn_5": neighbors.KNeighborsRegressor(n_neighbors=5),
        "knn_10": neighbors.KNeighborsRegressor(n_neighbors=10),
        "knn_15": neighbors.KNeighborsRegressor(n_neighbors=15),
    }
    
    sl = SuperLearner(library, task='regression', threshold=0.01, verbose=True)
    sl2 = SuperLearner(library, task='regression', threshold=0.01, meta_learner=linear_model.ElasticNetCV(alphas=np.arange(0.05, 15.0, 0.05), l1_ratio=np.arange(0.1, 1, 0.1), positive=True))
    
    
    print("Fitting...")
    sl.fit(X_train, y_train)
    sl2.fit(X_train, y_train)

    banana = True
    if banana:
        scores_test = []
        scores_train = []

        for estimator in library.values():
            estimator.fit(X_train, y_train)
            scores_train.append(estimator.score(X_train, y_train))
            scores_test.append(estimator.score(X_test, y_test))
        scores_train.append(sl.score(X_train, y_train))
        scores_test.append(sl.score(X_test, y_test))

        models = list(library.keys())
        print(models)

        fig, axs = plt.subplots(1,3)
        axs[0].bar(models, sl.weights, color='darkseagreen', width=0.5)
        axs[0].bar_label = [round(w, 2) for w in sl.weights]
        axs[0].set_title("Weights")

        models.append("SL")

        axs[2].bar(models, scores_train, color='blue', width=0.5)
        axs[2].set_title("Train")

        highest_score_index = scores_test.index(max(scores_test))
        colors = ['goldenrod' if i != highest_score_index else 'red' for i in range(len(models))]
        axs[1].bar(models, scores_test, color=colors, width=0.5)
        axs[1].set_title("Test")

        plt.show()


    #Print the weights
    for i, estimator in enumerate(library):
        print(estimator, ": ", sl.weights[i], " ", "Train: ", scores_train[i], " ", "Test: ", scores_test[i])
        
    banana = True
    if banana:
        scores_test[-1] = sl2.score(X_test, y_test)
        scores_train[-1] = sl2.score(X_train, y_train)

        models = list(library.keys())
        print(models)

        fig, axs = plt.subplots(1,3)
        axs[0].bar(models, sl2.weights, color='darkseagreen', width=0.5)
        axs[0].bar_label = [round(w, 2) for w in sl2.weights]
        axs[0].set_title("Weights")

        models.append("SL")

        axs[2].bar(models, scores_train, color='blue', width=0.5)
        axs[2].set_title("Train")

        highest_score_index = scores_test.index(max(scores_test))
        colors = ['goldenrod' if i != highest_score_index else 'red' for i in range(len(models))]
        axs[1].bar(models, scores_test, color=colors, width=0.5)
        axs[1].set_title("Test")

        plt.show()


    #Print the weights
    for i, estimator in enumerate(library):
        print(estimator, ": ", sl2.weights[i], " ", "Train: ", scores_train[i], " ", "Test: ", scores_test[i])
        
        
if __name__ == "__main__":
    main()
    