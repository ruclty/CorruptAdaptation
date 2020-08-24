# Codebase for "Generative Adversarial Imputation Networks (GAIN)"

Authors: Jinsung Yoon, James Jordon, Mihaela van der Schaar

Github Link: https://github.com/jsyoon0823/GAIN 

Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf

### Command inputs:
-   data_name: name of missing data
-   batch_size: batch size
-   hint_rate: hint rate
-   alpha: hyperparameter
-   iterations: iterations

### Example command

```shell
$ python main.py --data_name HTRU2_test_0.5 
--miss_rate: 0.2 --batch_size 128 --hint_rate 0.9 --alpha 100
--iterations 10000
```
