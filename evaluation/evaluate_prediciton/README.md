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
 - adapt_data: required, path of the adapt data
 - input_shape: required, input number of features
 - label_column: required, name of label column
 - epoch_space: required, training epoch 
 - l2: required, l2 regularization parameter
 - consistency: required, 'yes' or 'no' for using consistency loss or not  

Folder "params" contains examples, you can run the code using those parameter files directly, or write a self-defined parameter file to train a new dataset.

But for simple version, "consistency" and "adapt_data" is not required.

## Run
Run the code with the following command :
```sh
python evaluate_torch_consistency.py [parameter file]
```

For simple version please run evaluate_torch_simple.py.
