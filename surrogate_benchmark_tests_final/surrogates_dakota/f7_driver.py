# takes parameter and results files from Dakota and computes [function 7] valuations, writing them back to Dakota

import sys
import numpy as np

def f7(x):
    """
    f7(x) = (xᵀ A x)²,  A = diag(1,2,…,20),  x ∈ ℝ²⁰
    """
    x = np.asarray(x)
    assert x.shape == (30,), "x must be length-20"
    A_diag = np.arange(1, 31)
    quad = np.sum(A_diag * x**2)
    return quad**2

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
    x = read_params(sys.argv[1], 30)        # CHANGE THIS LINE
    y = f7(x)                               # CHANGE THIS LINE
    write_results(sys.argv[2], y)
