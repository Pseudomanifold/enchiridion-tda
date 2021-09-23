#!/usr/bin/env python3

import logging
import os
import sys

import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO)

# These parameters have been obtained using the `classification.py`
# script. Your mileage may vary, in particular if you choose to employ
# another kernel.
best_parameters = {
    'COLLAB': {'C': 0.01},
    'DD': {'C': 0.1},
    'FINGERPRINT': {'C': 0.01},
    'FRANKENSTEIN': {'C': 1.0},
    'IMDB-BINARY': {'C': 0.01},
    'IMDB-MULTI': {'C': 10.0},
    'MUTAG': {'C': 1000.0},
    'Mutagenicity': {'C': 10.0},
    'NCI1': {'C': 10.0},
    'NCI109': {'C': 10.0},
    'PTC_MR': {'C': 100.0},
    'PROTEINS': {'C': 1.0},
    'REDDIT-BINARY': {'C': 0.01},
    'REDDIT-MULTI-5k': {'C': 0.01}
}

if __name__ == '__main__':
    directories = sys.argv[1:]
    test_size   = 0.10

    for directory in directories:
        logging.info('Processing {}...'.format(directory))

        name       = os.path.basename(directory)
        filename_X = os.path.join(directory, 'Kernel.txt.gz')
        filename_y = os.path.join(directory, 'Labels.txt.gz')

        assert os.path.exists(filename_X), 'Missing file {}'.format(filename_X)
        assert os.path.exists(filename_y), 'Missing file {}'.format(filename_y)

        X = np.genfromtxt(filename_X)
        y = np.genfromtxt(filename_y)
        y = LabelEncoder().fit_transform(y)

        n, m = X.shape
        assert n == m, 'Kernel matrix is invalid'

        clf        = SVC(kernel='precomputed', probability=True)
        parameters = best_parameters.get(name, None)
        if not parameters:
            logging.warning('Using default parameters for data set {}'.format(name))
        else:
            logging.info('Setting parameters for data set {} to {}'.format(name, parameters))
            clf.set_params(**parameters)

        np.random.seed(42)

        mean_scores = []

        for i in range(10):

            scores = []

            sss = StratifiedShuffleSplit(n_splits = 10, test_size = test_size)
            for train, test in sss.split(np.zeros(n), y):

                X_train = X[train][:, train]
                y_train = y[train]
                X_test  = X[test][:, train]
                y_test  = y[test]

                clf.fit(X_train, y_train)
                y_pred   = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                scores.append(accuracy)
            
            print('{:2.2f}'.format(np.mean(scores) * 100.0))

        mean_scores.append(np.mean(scores))

    print('{:2.2f} += {:2.2}'.format(np.mean(mean_scores) * 100, np.std(mean_scores) * 100))
