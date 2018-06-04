#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import warnings

import numpy  as np
import pandas as pd

from sklearn.metrics         import accuracy_score
from sklearn.metrics         import average_precision_score
from sklearn.metrics         import f1_score
from sklearn.metrics         import precision_score
from sklearn.metrics         import recall_score
from sklearn.metrics         import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline        import make_pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.svm             import SVC

def find_hyperparameters(X, y, parameters):
    """
    Performs a grid search to find the best hyperparameters.
    """

    #grid = {
    #    'logisticregression__C'      : 10. ** np.arange(-5, 5),
    #    'logisticregression__penalty': ['l1', 'l2']
    #}

    svm = SVC(
        kernel      = 'precomputed',
        probability = True
    )

    clf = None

    if parameters['standardize']:
        clf = make_pipeline(
            StandardScaler(),
            svm
        )
    else:
        clf = make_pipeline(
            svm
        )

    clf.fit(X, y)
    return clf

    #grid_search = GridSearchCV(
    #    clf,
    #    param_grid   = grid,
    #    scoring      = 'average_precision',
    #    cv           = parameters['n_folds'],
    #    refit        = True,
    #)

    #grid_search.fit(X, y)

    # This returns the best estimator, which might either be
    # a classifier on its own or a pipeline.
    #return grid_search.best_estimator_, grid_search.best_score_

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
    print('  - Accuracy         : {:0.2f}'.format(accuracy))
    print('  - Average precision: {:0.2f}'.format(pr_auc))
    print('  - F1               : {:0.2f}'.format(f1))
    print('  - Precision        : {:0.2f}'.format(precision))
    print('  - Recall           : {:0.2f}'.format(recall))
    print('  - ROC AUC          : {:0.2f}'.format(roc_auc))

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

        best_clf = find_hyperparameters(
            X_train,
            y_train,
            parameters
        )

        ################################################################
        # Perform predictions on the best classifier
        ################################################################

        predict(
            name,
            X_test,
            y_test,
            best_clf
        )
