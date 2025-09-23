# takes parameter and results files from Dakota and computes [function 12] valuations, writing them back to Dakota

import sys
import numpy as np

def f12(x):
    """
    f12(x) = 1e6 · Π_{i=1..11} x_i^{α_i},  x ∈ ℝ²⁰
    α = [-0.00133172, -0.002270927, -0.00248546,
         -4.67, -4.671973, -0.00814,
         -0.008092, -0.005, -0.000909,
         -0.00088, -0.00119]
    """
    x = np.asarray(x)
    assert x.shape == (11,), "x must be length-20"
    alpha = np.array([
        -0.00133172, -0.002270927, -0.00248546,
        -4.67,       -4.671973,   -0.00814,
        -0.008092,   -0.005,      -0.000909,
        -0.00088,    -0.00119
    ])
    return 1e6 * np.prod(x[:11]**alpha)


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
    x = read_params(sys.argv[1], 11)        # CHANGE THIS LINE
    y = f12(x)                               # CHANGE THIS LINE
    write_results(sys.argv[2], y)
