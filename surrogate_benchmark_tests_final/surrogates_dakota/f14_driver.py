# takes parameter and results files from Dakota and computes [function 14] valuations, writing them back to Dakota

import sys
import numpy as np

def f14(x):
    """
    f14(x) = Σ_{i=1..20} [ α_i(x) ]²,  x ∈ ℝ²⁰
    α_i(x) = 420·x_i + (i−15)³
             + Σ_{j=1..20} v_{ij}[sin(log(v_{ij}))⁵ + cos(log(v_{ij}))⁵],
    v_{ij} = sqrt(j² + i/j)
    """
    x = np.asarray(x)
    assert x.shape == (30,), "x must be length-30"
    ndim = 30
    i = np.arange(1, ndim+1)
    j = i[:, None]
    k = i[None, :]
    v = np.sqrt(k**2 + j/k)
    trig = np.sin(np.log(v))**5 + np.cos(np.log(v))**5
    C = np.sum(v * trig, axis=1)
    α = 420.0*x + (i - 15.0)**3 + C
    return np.sum(α**2)

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
    y = f14(x)                               # CHANGE THIS LINE
    write_results(sys.argv[2], y)
