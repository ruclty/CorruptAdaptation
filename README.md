# DAGAN
DAGAN is a framework used in adaptive data augmentation for supervised learning over missing data. It extracts noise patterns from target data, and adapts the source data with the extracted target noise patterns while still preserving supervision signals in the source. Then, it patches the ML model by
retraining it on the adapted data, in order to better serve the target.

## Framework
As in the following figure, DAGAN consists of two connected GAN networks. The first GAN learns the noise pattern from the target, for target mask generation. The second GAN uses the learned target mask to augment the source data, for source data adaptation. The augmented source data can be used to retrain the ML model.
![avatar](https://github.com/ruclty/dagan/blob/master/figs/architecture.jpg)

## Paper and Data
For more details, please refer to our paper [Adaptive Data Augmentation for Supervised Learning over Missing Data](). Public datasets used in the paper can be downloaded from the [datasets page](https://github.com/ruc-datalab/dagan/tree/main/dataset).

## Quick Start
### Step1: Requirements
Before running the code, please make sure your Python version is above **3.6**.
Then install necessary packages by :
```sh
pip3 install -r requirements.txt
```

### Step2: Parameters
 You need to write a .json file as the configuration. The keyworks should include :

 - name: required, name of the output file 
 - source: required, path of the source data 
 - target: required, path of the target data 
 - target_mask: required, path of the target mask matrix
 - gen_model: required, neural network of the generator
 - normalize_cols: required, index of the numerical attributes normalized by simple-normalization 
 - gmm_cols: required, index of the numerical attributes normalized by GMM-normalization 
 - one-hot_cols: required, index of the categorical attributes encoded by one-hot encoding 
 - ordinal_cols:mrequired, index of the categorical attributes encoded by ordinal encoding 
 - epochs: required, number of training epochs 
 - steps_per_epoch: required, steps per epoch
 - rand_search: required, whether to search hyper-parameters randomly, yes or no ,
 - param: required if rand_search is 'no', hyper-parameter of the neural network  

Folder "code/params" contains examples, you can run the code using those parameter files directly, or write a self-defined parameter file to train a new dataset.

### Step3: Run
Run the code with the following command :
```sh
python code/train.py [parameter file] [gpu_id]
```
A example running command:
```sh
python code/train.py code/params/param-eyestate-MNAR 0
```

### Step4: Evaluation
Run the code with the following command :
```sh
python code/evaluate.py --train=[training file] --test=[test file] --label_col=[name of label column] --output=[output filename] --device=[gpu id]
```
A example running command:
```sh
python code/evaluate.py --train=ipums_adapt.csv --test=dataset/ipums/ipums_test.csv --label_col=movedin --output=ipums_result --device=0
```

## The Team
DAGAN was developed by Renmin University of China Phd student Tongyu Liu and grad student Yinqing Luo, under the supervision of Professor Ju Fan and Professor Xiaoyong Du.
