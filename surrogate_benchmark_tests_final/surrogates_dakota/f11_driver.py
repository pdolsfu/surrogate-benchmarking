# takes parameter and results files from Dakota and computes [function 11] valuations, writing them back to Dakota

import sys
import numpy as np

def f11(x):
    """
    f11(x) = Σ_{i=1..65} [y_i 
              − x₁e^(−x₅t_i)
              − x₂e^(−x₆(t_i−x₉)²)
              − x₃e^(−x₇(t_i−x₁₀)²)
              − x₄e^(−x₈(t_i−x₁₁)²)
             ]²,  x ∈ ℝ²⁰
    """
    x = np.asarray(x)
    assert x.shape == (11,), "x must be length-20"
    # data
    t = 0.1 * np.arange(65)
    y = np.array([
        1.366,1.191,1.112,1.013,0.991,0.885,0.831,0.847,0.786,0.725,0.746,
        0.679,0.608,0.655,0.616,0.606,0.602,0.626,0.651,0.724,0.649,0.649,
        0.694,0.644,0.624,0.661,0.612,0.558,0.533,0.495,0.500,0.423,0.395,
        0.375,0.372,0.391,0.396,0.405,0.428,0.429,0.523,0.562,0.607,0.653,
        0.672,0.708,0.633,0.668,0.645,0.632,0.591,0.559,0.597,0.625,0.739,
        0.710,0.729,0.720,0.636,0.581,0.428,0.292,0.162,0.098,0.054
    ])
    p = x[:11]
    a1,a2,a3,a4 = p[0],p[1],p[2],p[3]
    b1,b2,b3,b4 = p[4],p[5],p[6],p[7]
    c1,c2,c3    = p[8],p[9],p[10]
    M = (
        a1 * np.exp(-b1 * t)
      + a2 * np.exp(-b2 * (t - c1)**2)
      + a3 * np.exp(-b3 * (t - c2)**2)
      + a4 * np.exp(-b4 * (t - c3)**2)
    )
    return np.sum((y - M)**2)


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
    y = f11(x)                               # CHANGE THIS LINE
    write_results(sys.argv[2], y)
