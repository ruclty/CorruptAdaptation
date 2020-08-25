# Corrupt Domain Adaptation Evaluation

## Requirements
Before running the code, please make sure your Python version is above **3.5**.
Then install necessary packages by :
```sh
pip3 install -r requirements.txt
```

## Parameters
 You need to write a .json file as the configuration. The keywords should include :

 - name: required, name of the output file 
 - train_data: required, path of the train data 
 - test_data: required, path of the test data 
 - imputation: required, imputation method including "ZERO" or "MEAN"
 - input_shape: required, input number of features
 - label_column: required, name of label column
 - epoch_space: required, list of training epochs for grid search 
 - batch_size: required, list of training batch size for grid search
 - l2: required, list of l2 regularization parameter for grid search  

Folder "params" contains examples, you can run the code using those parameter files directly, or write a self-defined parameter file to train a new dataset.

## Run
Run the code with the following command :
```sh
python evaluate.py [parameter file]
```
