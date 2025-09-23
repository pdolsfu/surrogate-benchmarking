# takes parameter and results files from Dakota and computes [function 13] valuations, writing them back to Dakota

import sys
import numpy as np

def f13(x):
    """
    f13(x) = Σ_{i=1..14} a_i / x_i,  x ∈ ℝ²⁰
    a = [12842.275, 634.25, 634.25, 634.125,
         1268, 633.875, 633.75, 1267,
         760.05, 33.25, 1266.25, 632.875,
         394.46, 940.838]
    """
    x = np.asarray(x)
    assert x.shape == (14,), "x must be length-14"
    a = np.array([
        12842.275, 634.25, 634.25, 634.125,
        1268,      633.875, 633.75, 1267,
        760.05,    33.25, 1266.25,632.875,
        394.46,    940.838
    ])
    return np.sum(a / x[:14])

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
    x = read_params(sys.argv[1], 14)        # CHANGE THIS LINE
    y = f13(x)                               # CHANGE THIS LINE
    write_results(sys.argv[2], y)
