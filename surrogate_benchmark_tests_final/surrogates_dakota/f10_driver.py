# takes parameter and results files from Dakota and computes [function 10] valuations, writing them back to Dakota

import sys
import numpy as np

def f10(x):
    """
    f10(x) = Σ x_i² + [½ Σ x_i]² + [½ Σ x_i]⁴,  x ∈ ℝ²⁰
    """
    x = np.asarray(x)
    assert x.shape == (20,), "x must be length-20"
    term1 = np.sum(x**2)
    s = 0.5 * np.sum(x)
    return term1 + s**2 + s**4

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
    y = f10(x)                               # CHANGE THIS LINE
    write_results(sys.argv[2], y)
