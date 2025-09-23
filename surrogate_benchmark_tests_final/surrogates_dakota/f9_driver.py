# takes parameter and results files from Dakota and computes [function 9] valuations, writing them back to Dakota

import sys
import numpy as np

def f9(x):
    """
    f9(x) = xᵀA x − 2·x₁,
    A tridiagonal with [1,2,…,2] on diag and -1 off–diag,  x ∈ ℝ²⁰
    """
    x = np.asarray(x)
    assert x.shape == (20,), "x must be length-20"
    # main diagonal
    main = x[0]**2 + 2*np.sum(x[1:]**2)
    # off‐diagonal A_{i,i+1} = A_{i+1,i} = -1
    off = -2*np.sum(x[:-1]*x[1:])
    return main + off - 2*x[0]

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
    x = read_params(sys.argv[1], 20)        # CHANGE THIS LINE
    y = f9(x)                               # CHANGE THIS LINE
    write_results(sys.argv[2], y)
