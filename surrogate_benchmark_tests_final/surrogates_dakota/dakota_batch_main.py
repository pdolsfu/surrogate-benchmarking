# script to run all the dakota methods on the function from the problem set specified in line 60

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import re
import os
import subprocess
from func_defs import *

# Computer R^2, RAAE, and RMAE
def compute_accuracy(y_true, y_surrogate):
    metrics = []
    r2 = r2_score(y_true, y_surrogate)

    mae = np.mean(np.abs(y_true - y_surrogate))
    std_y = np.std(y_true)
    raae = mae / std_y

    max_ae = np.max(np.abs(y_true - y_surrogate))
    rmae = max_ae / std_y 

    return {'RAAE': raae, 'RMAE': rmae, 'R2': r2}

# Writing the error metrics to the file in .dat format
def write_accuracy(metric, model_type, time, file_path):
    """
    Append one row to a tab-delimited .dat file with columns:
      time    R2    RAAE    RMAE    model_type
    All floats are in scientific notation with 3 decimal places.
    """
    # Check if we need to write the header
    write_header = not os.path.exists(file_path) or os.stat(file_path).st_size == 0

    with open(file_path, 'a', newline='') as f:
        if write_header:
            f.write("time\tR2\tRAAE\tRMAE\tmodel_type\n")
        # now write one row
        f.write(
            f"{time:.3e}\t"
            f"{metric['R2']:.3e}\t"
            f"{metric['RAAE']:.3e}\t"
            f"{metric['RMAE']:.3e}\t"
            f"{model_type}\n"
        )

# Extracts the time from .out file, which is always on the third last line of the file
def extract_CPU_time(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        target_line = lines[-2].strip()  # this should always be the line with the CPU clock time       
        match = re.search(r'Total CPU\s*=\s*([\d\.]+)', target_line)
        if match:
            total_time = float(match.group(1))
            return total_time
        else:
            raise ValueError("Could not find 'Total CPU' value in the expected format.")


def main(ndim=10, function_number=1):  # this is the only line of that needs modification from the user
    # Set file path for the results generated for function x
    base_dir = r"\your_path.....\dakota_benchmark_results" 
    folder_name = f"results{function_number}.dat"
    abs_path = os.path.join(base_dir, folder_name)    
    open(abs_path, "w").close()

    # Initialize variables and vectors
    func_name = f"true_f{function_number}"
    f_class  = globals()[func_name]
    function = f_class(ndim=ndim)     

    list = ["gaussian_process dakota", "mars", "neural_network", "radial_basis", "polynomial linear", "polynomial quadratic", "polynomial cubic"]
    for model_type in list:                  
        print(f"Running this method in Dakota: {model_type}")
    
        # Iterate through the list of methods in Dakota
        with open(f"f{function_number}_surrogate.in", "r") as f:
            lines = f.readlines()
        lines[17] = f"    {model_type}\n"                                  # The line with the method specification
        with open(f"f{function_number}_surrogate.in", "w") as f:
            f.writelines(lines)

        # Running the nth Dakota method with terminal command
        subprocess.run(f"dakota -i f{function_number}_surrogate.in -o f{function_number}_surrogate.out", shell=True, check=True)

        # Extract test inputs and surrogate predictions from surrogate.dat file
        df = pd.read_csv(f'f{function_number}_surrogate.dat', sep=r'\s+', skiprows=1)      
        xtest = df.iloc[:, 2:(ndim+2)].values
        y_pred = df.iloc[:, (ndim+2)].values                               # surrogate predictions are at the far right of the .dat file

        # Evaluate the surrogate accuracy
        y_true = function(xtest)                                           # evaluate true function    
        metrics = compute_accuracy(y_true, y_pred)                         # compute accuracy metrics   
        time = extract_CPU_time(f"f{function_number}_surrogate.out")       # extracts CPU clock time from .out file
        write_accuracy(metrics, model_type, time, abs_path)                # append the new metrics to the .dat file

if __name__ == "__main__":
    main()