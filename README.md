# Higgs Boson Machine Learning Project

In this file, you can find useful information about the code of ML project 1, how to run it, etc. 

## 1. Where to put the dataset
In general, you can simply place both the train.csv and test.csv files in a folder titled 'data' and put this folder in the same directory as the Python code files. Please note that the zipped 'data' folder is already available in this directory. However, please make sure that you unzip this folder (and the csv files in it) before running the code. 

## 2. How to run the code
In order to run the code (and reproduce our Kaggle results), all you need is to run the `run.py` file. 

## 3. Where to find the outputs (i.e. final predictions)
After the main code `run.py` is run, the results are automatically saved in 'outputs' folder under the name 'output.csv'. This file can directly be uploaded on Kaggle.

## 4. About the files in this directory
- `run.py`
This script contains the main code that produces final Kaggle predictions.

- `implementations.py`
The implementation of 6 mandatory algorithms (as listed in project1 description) is found in this file. 

- `helpers.py`
This script contains the (helping) functions used in reading data, writing final results, etc

- `validation.py`
In this file, the validation functions such as cross-validation (data_split) are found. 

* `run_regression.ipynb`
This jupyter notebook does the same job as the 'run.py'(see above). 

