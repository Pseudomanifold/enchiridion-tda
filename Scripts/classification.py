#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import warnings

import numpy  as np
import pandas as pd

from sklearn.base            import clone
from sklearn.metrics         import accuracy_score
from sklearn.metrics         import average_precision_score
from sklearn.metrics         import f1_score
from sklearn.metrics         import precision_score
from sklearn.metrics         import recall_score
from sklearn.metrics         import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline        import make_pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.svm             import SVC

class KernelGridSearchCV:
    """
    A simple class for performing a grid search for kernel matrices with
    a cross-validation strategy. At present, the class interface follows
    the default interface of `scikit-learn`. However, the class is *not*
    yet inheriting from any base class.
    """

    def __init__(self, clf, param_grid, n_folds, random_state = None, refit = True):
        self.clf_             = clf
        self.grid_            = param_grid
        self.n_folds_         = n_folds
        self.random_state_    = random_state
        self.refit_           = refit
        self.best_estimator_  = None
        self.best_score_      = None

    def fit(self, X, y):
        cv = StratifiedKFold(
                n_splits     = self.n_folds_,
                shuffle      = True,
                random_state = self.random_state_
        )

        grid = ParameterGrid(self.grid_)
        for parameters in grid:
            clf = self.clf_
            clf.set_params(**parameters)

            scores = []
            for train, test in cv.split(np.zeros(len(y)), y):
                X_train = X[train][:, train]
                y_train = y[train]
                X_test  = X[test][:, train]
                y_test  = y[test]

                clf.fit(X_train, y_train)
                y_pred_proba = clf.predict_proba(X_test)

                # TODO: should make this configurable in order to
                # support more scoring functions
                ap = average_precision_score(y_test, y_pred_proba[:,1], average='weighted')
                scores.append(ap)

            score = np.mean(scores)
            if self.best_score_ is None or score > self.best_score_:
                self.best_estimator_ = clone(clf)
                self.best_score_     = score
                self.best_params_    = parameters

def find_hyperparameters(X, y, parameters):
    """
    Performs a grid search to find the best hyperparameters.
    """

    grid = {
        'svc__C': 10. ** np.arange(-2, 5),
    }

    svm = SVC(
        kernel      = 'precomputed',
        probability = True
    )

    if parameters['standardize']:
        clf = make_pipeline(
            StandardScaler(),
            svm
        )
    else:
        clf = make_pipeline(
            svm
        )

    grid_search = KernelGridSearchCV(
        clf,
        param_grid   = grid,
        n_folds      = parameters['n_folds'],
        refit        = True,
        random_state = 42
    )

    grid_search.fit(X,y)

    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_

def predict(name, X, y, clf):
    """
    Performs predictions on a data set using a given classifier and
    reports several performance measures.
    """

    y_pred_proba = clf.predict_proba(X)
    y_pred       = clf.predict(X)

    accuracy     = accuracy_score(y, y_pred)
    pr_auc       = average_precision_score(y, y_pred_proba[:,1], average='weighted')
    f1           = f1_score(y, y_pred, average='weighted')
    precision    = precision_score(y, y_pred, average='weighted')
    recall       = recall_score(y, y_pred, average='weighted')
    roc_auc      = roc_auc_score(y, y_pred_proba[:,1])

    print('---')
    print(name)
    print('---')
    print('  - Accuracy         : {:0.4f}'.format(accuracy))
    print('  - Average precision: {:0.4f}'.format(pr_auc))
    print('  - F1               : {:0.4f}'.format(f1))
    print('  - Precision        : {:0.4f}'.format(precision))
    print('  - Recall           : {:0.4f}'.format(recall))
    print('  - ROC AUC          : {:0.4f}'.format(roc_auc))

    sys.stdout.flush()

if __name__ == '__main__':

    # FIXME: make configurable
    standardize = False
    folds       = 4
    test_size   = 0.20

    parameters = {
        'standardize': standardize, # Flag indicating whether standardization should be used
        'n_folds'    : folds,       # Number of folds
        'test_size'  : test_size    # Size of the test data set
    }

    # FIXME: make configurable
    directories = sys.argv[1:]

    for directory in directories:
        logging.info('Processing {}...'.format(directory))

        name       = os.path.basename(directory)
        filename_X = os.path.join(directory, 'Kernel.txt')
        filename_y = os.path.join(directory, 'Labels.txt')

        assert os.path.exists(filename_X), 'Missing file {}'.format(filename_X)
        assert os.path.exists(filename_y), 'Missing file {}'.format(filename_y)

        X = np.genfromtxt(filename_X)
        y = np.genfromtxt(filename_y)
        y = np.where(y > 0.0, 1, 0)   # Make the class labels binary as I do not
                                      # want to tackle multi-class clasification
                                      # for now.

        n, m = X.shape
        assert n == m, 'Kernel matrix is invalid'

        ################################################################
        # Hyperparameter search
        ################################################################

        logging.info('Finding best hyperparameters with test size = {:0.2f}...'.format(test_size))

        sss                         = StratifiedShuffleSplit(n_splits = 1, random_state = 42)
        [(train, test)]             = sss.split(np.zeros(n), y)

        X_train = X[train][:, train]
        y_train = y[train]
        X_test  = X[test][:, train]
        y_test  = y[test]

        best_clf, best_score, best_params = find_hyperparameters(
            X_train,
            y_train,
            parameters
        )

        print(best_params)

        # Refit to the training data set because we cannot use the
        # classifier otherwise.
        best_clf.fit(X_train, y_train)

        ################################################################
        # Perform predictions on the best classifier
        ################################################################

        predict(
            name,
            X_test,
            y_test,
            best_clf
        )
