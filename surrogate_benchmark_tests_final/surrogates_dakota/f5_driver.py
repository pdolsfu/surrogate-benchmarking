# takes parameter and results files from Dakota and computes [function 5] valuations, writing them back to Dakota

import sys
import numpy as np

def f5(x):
    """
    f5(x) = sum_{i=1}^5 [
        (x_i + 10*x_{i+5})^2
      + 5*(x_{i+10} - x_{i+15})^2
      + (x_{i+5} - 2*x_{i+10})^4
      + 10*(x_i - x_{i+15})^4
    ],  x ∈ ℝ²⁰
    """
    x = np.asarray(x)
    assert x.shape == (20,), "x must be length-20"
    out = 0.0
    for i in range(5):
        xi   = x[i]
        xi5  = x[i + 5]
        xi10 = x[i + 10]
        xi15 = x[i + 15]
        term1 = (xi + 10*xi5)**2
        term2 = 5*(xi10 - xi15)**2
        term3 = (xi5 - 2*xi10)**4
        term4 = 10*(xi - xi15)**4
        out += term1 + term2 + term3 + term4
    return out
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
    y = f5(x)                               # CHANGE THIS LINE
    write_results(sys.argv[2], y)
