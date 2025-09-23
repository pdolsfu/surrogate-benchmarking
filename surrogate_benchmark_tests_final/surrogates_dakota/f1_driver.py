# takes parameter and results files from Dakota and computes true function valuations, writing them back to Dakota

import sys
import numpy as np

def f1(x):
    term1 = (x[0] - 1)**2
    term2 = (x[9] - 1)**2
    term3 = 0
    
    for i in range(9):  # the term with the riemann sum
        term3 += (10 - (i + 1)) * (x[i]**2 - x[i+1])**2

    fx = term1 + term2 + 10 * term3
    return fx

# reading a set of values
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
    x = read_params(sys.argv[1], 10)
    y = f1(x)
    write_results(sys.argv[2], y)
