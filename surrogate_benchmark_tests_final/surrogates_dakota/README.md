### Dakota Code

# Run this terminal command to install the necessary Python libraries: pip install numpy scikit-learn pandas

# Dakota 6.21, used in this test, can be downloaded here: https://snl-dakota.github.io/docs/6.21.0/users/setupdakota.html#download-dakota

# The surrogate model training process in Dakota is not significantly different than that of Python, except that each function required its own Dakota .in file for specifying several pieces of information of the function, such as bound constraints, output file name, and number of dimensions. Unlike the Python code, the Dakota code was not automated to run all the models for a given function. Instead, The line specifying the method needs to be manually changed each time a different method is to be used. In addition, Dakota utilizes a Python driver for a given function to begin true function evaluations, which is called individually for each pair of inputs. Additionally, Dakota version 6.21 does not recognize the training_data_file keyword, which would allow externally loading of surrogate model training files. This would moderately reduce runtime as LHS training sample generation would only need to be done once for each function test.

# dakota_batch_main.py runs the function specified in the code. ".in" files are the Dakota files responsible for the surrogate model training, and the corresponding xx.driver files are responsible for the true function evaluations. The ".out", ".dat", ".rst", and ".13" files can be ignored. The results of the tests can be found in dakota_benchmark_results

# The method documentation is very straight forward, as the methods are predefined and require only modifying a single line of code to indicate which method to run. The list of global surrogate models in Dakota and their documentation can be found here: https://snl-dakota.github.io/docs/6.22.0/users/usingdakota/reference/model-surrogate-global.html
