"""
This script processes and evaluates a set of result files from an ensemble training over different gradient descent methods provided by optax.
It identifies the best result file based on a computed metric.

The script performs the following steps:
1. Lists all result files in the specified results folder.
2. Iterates through each result file, loading and evaluating the results.
3. Computes a metric to evaluate the results.
4. Identifies and prints the filename of the best result based on the computed metric.

Constants:
- `RESULTS_FOLDER`: The folder containing the result files.
- `RESULTS_FILES`: A list of filenames in the results folder.
- `COUNT`: An initial large value used for comparison to find the best result.
- `FILENAME_BEST`: The filename of the best result found.

Functions:
- `listdir`: Lists all entries in the specified directory.
- `isfile`: Checks if the specified path is a file.
- `join`: Joins one or more path components.
- `np.nan_to_num`: Replaces NaNs with zero and infinities with large finite numbers.
- `np.loadtxt`: Loads data from a text file.
- `np.sqrt`: Computes the square root.
- `np.sum`: Sums array elements.
- `np.square`: Squares each element in the array.
- `np.isnan`: Checks if a value is NaN.
- `np.isneginf`: Checks if a value is negative infinity.
- `np.isposinf`: Checks if a value is positive infinity.
"""

from os import listdir
from os.path import isfile, join
import numpy as np

RESULTS_FOLDER = "results/inference_ensemble_sgd/"
RESULTS_FILES = [f for f in listdir(RESULTS_FOLDER) if isfile(join(RESULTS_FOLDER, f))]

COUNT = 1e20
print(RESULTS_FILES)
FILENAME_BEST = ""
for filename in RESULTS_FILES:
    results = np.nan_to_num(np.loadtxt(RESULTS_FOLDER + filename)) / 1e8
    val = np.sqrt(np.sum(np.square(results[1, :] - results[2, :]))) / np.sqrt(
        np.sum(np.square(results[2, :]))
    )

    if (
        val < COUNT
        and not np.isnan(val)
        and not np.isneginf(val)
        and not np.isposinf(val)
    ):
        COUNT = val
        FILENAME_BEST = filename


print(FILENAME_BEST)
