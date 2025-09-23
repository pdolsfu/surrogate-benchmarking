# takes parameter and results files from Dakota and computes [function 4] valuations, writing them back to Dakota

import sys
import numpy as np

def f4(x):    
    x = np.asarray(x)
    assert x.shape == (20, ), "Each input vector must be 20-dimensional"
    print(f"still running don't worry")
    out = 0
    for i in range(5):
        xi   = x[i]
        xi5  = x[i + 5]
        xi10 = x[i + 10]
        xi15 = x[i + 15]

        term1 = 100.0 * (xi**2  + xi5)**2
        term2 = (xi - 1)**2
        term3 =  90.0 * (xi10**2 + xi15)**2
        term4 = (xi10 - 1)**2
        term5 =  10.1 * ((xi5  - 1)**2 + (xi15 - 1)**2)
        term6 =  19.8 * (xi5  - 1) * (xi15 - 1)

        out += (term1 + term2 + term3 + term4 + term5 + term6)
        
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
    y = f4(x)                               # CHANGE THIS LINE
    write_results(sys.argv[2], y)
