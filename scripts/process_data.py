import csv
import numpy as np
import cPickle
import ipdb

file_path = 'PathSolutions.txt'

# Count the number of samples
row_count = 0
row_width = 0
all_rows = []
with open(file_path, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        if row_count == 0:
            row_width = len(row)
        row_count = row_count + 1

label_count = 12
feature_count = row_width - label_count

# Load the samples into feature and target numpy arrays
X = np.zeros((row_count, feature_count))
y = np.zeros((row_count, label_count))
row_idx = 0
with open(file_path, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        for col_idx in xrange(feature_count):
            X[row_idx, col_idx] = float(row[col_idx])
        assert(np.sum(X[row_idx]) != 0)

        for col_idx in xrange(label_count):
            y[row_idx, col_idx] = float(row[feature_count + col_idx])
        assert(np.sum(y[row_idx]) != 0)

        row_idx = row_idx + 1

# Permute the data
np.random.seed(1)
permuted_idxs = np.random.permutation(X.shape[0])
X = X[permuted_idxs]
y = y[permuted_idxs]

# Save the data
pklfile = open('path_solutions.pkl', 'wb')
cPickle.dump(X, pklfile)
cPickle.dump(y, pklfile)
