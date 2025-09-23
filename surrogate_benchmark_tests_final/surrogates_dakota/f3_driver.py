# takes parameter and results files from Dakota and computes [function 3] valuations, writing them back to Dakota

import sys
import numpy as np

def f3(x):
    x = np.asarray(x)
    assert x.shape == (20, ), "Each input vector must be 20-dimensional"

    xi = x[:10]         # First 20 components
    xi_plus_10 = x[10:] # Last 20 components

    term1 = 100 * (xi - xi_plus_10)**2
    term2 = (xi - 1)**2

    return np.sum(term1 + term2)

# reading the set of values
def read_params(filename, n_variables):
    with open(filename) as f:
        lines = f.readlines()

    x = []  # List to store numeric values for variables

    for i, line in enumerate(lines):
        if 'variables' in line:  # Look for the 'variables' section
            for j in range(n_variables):
                x_line = lines[i + j + 1].split()  # Get the next line (after 'variables')
                x.append(float(x_line[0]))  # Extract only the numeric value (x1) and add to the array
            break
    return x     

# writing results
def write_results(filename, value):
    with open(filename, 'w') as f:
        f.write(f"{value:.6f}\n")

if __name__ == "__main__":
    x = read_params(sys.argv[1], 20)        # CHANGE DEPENDING ON FUNCTION DIMENSIONALITY
    y = f3(x)
    write_results(sys.argv[2], y)
