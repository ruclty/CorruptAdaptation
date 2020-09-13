# input argv dataname testname roundname
import numpy as np
import pandas as pd

from torch import nn
from torch import optim
from skorch import NeuralNetBinaryClassifier
import sys
import argparse
import json
import multiprocessing
import os
import time
from pandas.api.types import is_object_dtype


class MyModule(nn.Module):
    def __init__(self, shape=14, nonlin=nn.ReLU()):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(shape, 50)
        self.nonlin = nonlin
        # self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(50, 25)
        self.output = nn.Linear(25, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        # X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.sigmoid(self.output(X))
        return X


def feature_encoder(label_data):
    # transform categorical columns into numerical columns
    from sklearn.preprocessing import LabelEncoder
    from pandas.api.types import is_object_dtype
    label_con_data = label_data.copy()
    gens = {}
    for column in label_data.columns:
        if is_object_dtype(label_data[column]):
            gen_le = LabelEncoder()
            gen_labels = gen_le.fit_transform(list(label_data[column]))
            label_con_data.loc[:, column] = gen_labels  # to label from 0
            gens[column] = gen_le  # save the transformer to inverse
    # return a DataFrame
    return label_con_data, gens


# the model KerasClassifier needs
def create_model(shape=14, optimizer__weight_decay=0):
    # binary classification using MLP
    model = NeuralNetBinaryClassifier(
        MyModule(shape=shape),
        batch_size=-1,
        train_split=None,
        # default: lr=0.01,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
        criterion=nn.BCELoss,
        optimizer=optim.Adam,
        optimizer__weight_decay=optimizer__weight_decay,
        device="cuda")
    # criterion:torch criterion (class, default=torch.nn.BCEWithLogitsLoss)
    # threshold:float (default=0.5)
    #
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=[mf1])
    return model


def evaluation(y_true, y_pred, y_prob, threshold, parameters):
    # y_true is list of true labels
    # y_pred is list of predicted labels
    # y_prob is list of probabilities
    # threshold is the number by which we transform probability to one label
    # parameters is the parameters of the model
    from sklearn import metrics
    f1_score = metrics.f1_score(y_true, y_pred)  # exert the same importance on precision and recall
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    balanced_f1_score = metrics.f1_score(y_true, y_pred, average='weighted')
    balanced_precision = metrics.precision_score(y_true, y_pred, average='weighted')
    balanced_recall = metrics.recall_score(y_true, y_pred, average='weighted')
    return [recall, precision, f1_score, balanced_recall, balanced_precision, balanced_f1_score, threshold, parameters]


def result_sum(mlp_val, name):
    # name is the name of output file
    val_result = pd.DataFrame({'Neural Network': mlp_val},
                              index=['recall', 'precision', 'f1_score', 'balanced recall', 'balanced precision',
                                     'balanced f1_score', 'threshold', 'best_parameters'])

    val_result.to_csv(name, encoding='utf_8_sig')
    print('the results are in %s' % (name))


def thread_run(config, train_data, test_data):
    data = pd.concat([train_data, test_data], keys=['train', 'test'])
    
    # drop unnecessary columns
    try:
        data = data.drop(columns=['Unnamed: 0'])
    except:
        pass
    try:
        data = data.drop(columns=['Unnamed: 0.1'])
    except:
        pass
    try:
        data = data.drop(columns=['Unnamed: 0.1.1'])
    except:
        pass

    # transform the categorical columns into numerical columns
    featured_con_data, gens = feature_encoder(data)

    # split train and test from data
    label_column = config["label_column"]
    train_data = featured_con_data.loc['train']
    test_data = featured_con_data.loc['test']
    X_train = train_data.drop(axis=1, columns=[label_column])
    train_y = train_data[label_column]
    train_y = train_y.astype(np.float32)
    X_test = test_data.drop(axis=1, columns=[label_column])
    test_y = test_data[label_column]
    test_y = test_y.astype(np.float32)
    
    # scaling
    from sklearn import preprocessing
    scaled_test_x = preprocessing.scale(X_test)
    scaled_test_x = scaled_test_x.astype(np.float32)
    print('scaled test x shape:', scaled_test_x.shape[1])
    scaled_train_x = preprocessing.scale(X_train)
    scaled_train_x = scaled_train_x.astype(np.float32)
    print('scaled train x shape:', scaled_train_x.shape[1])
    
    # classfier training
    from sklearn.model_selection import GridSearchCV
    from sklearn import metrics

    # neural network
    # create the model
    model = create_model(shape=config["input_shape"][0]).initialize()
    # tune the parameters through cross-validation, evaluate by scoring='f1’
    # batch_size = config["batch_size"]
    epochs = config["epoch_space"]
    l2 = config["l2"]
    # shape = config["input_shape"]
    parameters_1 = dict(max_epochs=epochs, optimizer__weight_decay=l2)
    grid = GridSearchCV(model, param_grid=parameters_1, cv=5, scoring='f1',
                        verbose=2)
    print('Neural network model training begins...')
    grid_result = grid.fit(scaled_train_x, train_y)
    best_mlp = grid_result.best_estimator_
    best_mlp.fit(scaled_train_x, train_y)
    print('best parameters:', grid.best_params_)
    mlp_para = grid_result.best_params_
    mlp_thres = "default"
    # predict label on test set
    mlp_prob_test_y = best_mlp.predict(scaled_test_x)
    mlp_pred_test_y = best_mlp.predict(scaled_test_x)
    print('Neural network model training ends.')
    # evaluate
    mlp_val = evaluation(test_y, mlp_pred_test_y, mlp_prob_test_y, mlp_thres, mlp_para)

    # 输出格式化的结果
    result_sum(mlp_val, config["name"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', help='a json config file')
    args = parser.parse_args()

    with open(args.configs) as f:
        configs = json.load(f)

    for config in configs:

        train_data = pd.read_csv(config["train_data"])
        test_data = pd.read_csv(config["test_data"])

        job = multiprocessing.Process(target=thread_run, args=(config, train_data, test_data))
        time.sleep(3)
        job.start()
        job.join()

