# input argv dataname testname roundname
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from keras import models
from keras import layers
import keras.backend as K
from keras import optimizers
from keras.wrappers import scikit_learn
from keras import backend as K
import sys
import argparse
import json
import multiprocessing


def mf1(y_true, y_pred):

    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def feature_encoder(label_data):
    # transform catecorical columns into numerical columns
    from sklearn.preprocessing import LabelEncoder
    from pandas.api.types import is_object_dtype
    label_con_data = label_data.copy()
    for column in label_data.columns:
        if is_object_dtype(label_data[column]):
            gen_le = LabelEncoder()
            gen_labels = gen_le.fit_transform(list(label_data[column]))
            label_con_data[column] = gen_labels
    # return a DataFrame
    return label_con_data


# the model KerasClassifier needs
def create_model(l2, shape):
    # binary classification using MLP
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(shape,), kernel_regularizer=regularizers.l2(l2)))
    # model.add(Dropout(0.5))
    model.add(Dense(25, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[mf1])
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

    val_result.to_csv(name,encoding='utf_8_sig')
    print('the results are in %s' % (name))


def thread_run(config, train_data, test_data):
    data = pd.concat([train_data, test_data], keys=['train', 'test'])
    # imputation
    if config["imputation"]=="ZERO":
        data = data.fillna(0)
    elif config["imputation"]=="MEAN":
        for column in data.columns:
            if data[column].isnull().sum() > 0:
                if is_object_dtype(data[column]):
                    constant = data[column].mode()
                    data[column] = data[column].fillna(constant)
                else:
                    constant = data[column].mean()
                    data[column] = data[column].fillna(constant)
    
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
    featured_con_data = feature_encoder(data)
    
    # split train and test from data
    label_column = config["label_column"]
    train_data = featured_con_data.loc['train']
    test_data = featured_con_data.loc['test']
    X_train = train_data.drop(axis=1, columns=[label_column])
    train_y = train_data[label_column]
    X_test = test_data.drop(axis=1, columns=[label_column])
    test_y = test_data[label_column]
    
    # scaling
    from sklearn import preprocessing
    scaled_test_x = preprocessing.scale(X_test)
    print('scaled test x shape:', scaled_test_x.shape[1])
    scaled_train_x = preprocessing.scale(X_train)
    print('scaled train x shape:', scaled_train_x.shape[1])
    
    # classfier training
    from sklearn.model_selection import GridSearchCV
    from sklearn import metrics

    # neural network
    # create the model
    model = scikit_learn.KerasClassifier(build_fn=create_model, verbose=0)
    # tune the parameters through cross-validation, evaluate by scoring='f1’
    batch_size = config["batch_size"]
    epochs = config["epoch_space"]
    l2 = config["l2"]
    shape = config["input_shape"]
    parameters_1 = dict(batch_size=batch_size, epochs=epochs, l2=l2, shape=shape)
    grid = GridSearchCV(model, param_grid=parameters_1, cv=5, scoring='f1',
                        verbose=2)
    print('Neural network model training begins...')
    grid_result = grid.fit(scaled_train_x, train_y)
    best_mlp = grid_result.best_estimator_
    best_mlp.fit(scaled_train_x, train_y)
    print('best parameters:', grid.best_params_)
    mlp_para = grid_result.best_params_
    # find the best threshold using f1 score
    mlp_prob_train_y = best_mlp.predict_proba(scaled_train_x)[:, 1]
    precision, recall, threshold = metrics.precision_recall_curve(train_y, mlp_prob_train_y)
    f1 = (2 * precision * recall) / (precision + recall)
    f1 = np.nan_to_num(f1)
    thres = threshold[np.argmax(f1)]
    print('best threshold:', thres)
    mlp_thres = thres
    # predict label on test set
    mlp_prob_test_y = best_mlp.predict(scaled_test_x)
    mlp_pred_test_y = (mlp_prob_test_y > thres) + 0
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
    try:
        os.mkdir("expdir")
    except:
        pass

    for config in configs:
        path = "expdir/"+config["name"]+"/"
        try:
            os.mkdir("expdir/"+config["name"])
        except:
            pass
        train_data = pd.read_csv(config["train_data"])
        test_data = pd.read_csv(config["test_data"])

        job = multiprocessing.Process(target=thread_run, args=(config, train_data, test_data))
        job.start()

