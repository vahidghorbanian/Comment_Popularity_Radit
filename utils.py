import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor as dt
from sklearn.ensemble import RandomForestRegressor as rfr
import os.path
from numpy import genfromtxt
import matplotlib.pyplot as plt
import json


# Initialization
random_state = 0
max_iter = 1000
tol = 1e-5
fit_intercept = True
normalize = False

# Load data
def load_data(file_name):

    file = open(file_name)
    data = pd.DataFrame(data=json.load(file))
    data = data[['text', 'children', 'controversiality', 'is_root', 'popularity_score']]
    data['is_root'] = 1*data['is_root']
    num_samples = len(data)
    features = list(data.columns[:-1])
    target = data.columns[-1]
    print('total number of samples: ', num_samples)
    print('main features name:', features)
    print('target name:', target)
    return data, features, target


# split data
def split_data(data, type='given', validation_size=.2, test_size=0.1):
    if type == 'given':
        X_train = data.loc[0:10000-1, data.columns[:-1]]
        y_train = data.loc[0:10000-1, data.columns[-1]]
        X_validation = data.loc[10000:11000-1, data.columns[:-1]]
        y_validation = data.loc[10000:11000-1, data.columns[-1]]
        X_test = data.loc[11000:12000, data.columns[:-1]]
        y_test = data.loc[11000:12000, data.columns[-1]]
    else:
        num_sample = len(data)
        num_test = np.floor(test_size*num_sample)
        num_validation = np.floor(validation_size*num_sample)
        X = data.loc[:, data.columns[:-1]]
        y = data.loc[:, data.columns[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                        test_size=num_validation/(num_sample-num_test),
                                                                        random_state=random_state)
    print('length of training set: ', len(X_train))
    print('length of validation set: ', len(X_validation))
    print('length of test set: ', len(X_test))
    return {'type': 'training', 'X': X_train, 'y': y_train}, \
           {'type': 'validation', 'X': X_validation, 'y': y_validation},\
           {'type': 'test', 'X': X_test, 'y': y_test}


def high_rank_features(data, num_feature=160):
    # type = data['type']
    # txt = data['X']['text'].values
    type = 'train'
    txt = data[:, 0]
    print('\nlooking for top ' +str(num_feature)+ ' high rank text features in '+type+' set ...')
    for i in np.arange(0, len(txt), 1):
        txt[i] = txt[i].lower()
    words = txt.sum().split()
    words_unique = np.unique(words)
    print('\ntotal number of words in '+type+' set: ', len(words))
    print('total number of unique words in '+type+' set: ', len(words_unique), '\n')
    if os.path.exists('count.csv')==True:
        count = genfromtxt('count.csv', delimiter=',')
    else:
        count = np.zeros((len(words_unique), 1))
        for i in np.arange(0, len(words_unique), 1):
            count[i] = words.count(words_unique[i])
        np.savetxt('count.csv', count, delimiter=",")
    feature_name = []
    feature_count = []
    count = list(count)
    words_unique = list(words_unique)
    for i in np.arange(0, num_feature, 1):
        idx = np.argmax(count)
        feature_name.append(words_unique[idx])
        feature_count.append(count[idx])
        count.pop(idx)
        words_unique.pop(idx)
    return feature_name, feature_count, len(words)


def add_total_word_count(data, num_total_words):
    type = data['type']
    text = data['X']['text'].values
    print('Calculating text features counts in ' + type + ' set ...')
    count = []
    for i in np.arange(0, len(text), 1):
        # count.append(len(text[i].split())/num_total_words)
        count.append(len(text[i].split()))
    data['X']['text'] = text
    data['X']['count'] = count
    return data


def add_txt_features(data, feature_name):
    text = data['X']['text'].values
    features_count = np.zeros((len(text), len(feature_name)))
    for j, f in enumerate(feature_name):
        for i in np.arange(0, len(text), 1):
            features_count[i, j] = text[i].split().count(feature_name[j])
    print('Adding features to ' + data['type'] + ' set ...')
    for i, f in enumerate(feature_name):
        data['X'][f] = features_count[:, i]
    return data


def add_interaction_terms(data, order=2, include_bias=False):
    col1 = list(data['X'].columns)
    values = data['X'].values
    poly = PolynomialFeatures(order)
    print('Adding polynomial features to ' + data['type'] + ' set ...')
    if include_bias==True:
        poly_features = poly.fit_transform(data['X'].values[:, 1:])
        col2 = []
        for i in np.arange(len(col1)-1, np.shape(poly_features)[1], 1):
            col2.append('f'+str(i))
        col2[0] = 'bias'
        col = np.hstack([col1[0], col2[0], col1[1:], col2[1:]])
        val = np.vstack([values[:, 0], poly_features[:, 0]]).T
        val = np.hstack([val, poly_features[:, 1:]])
        data['X'] = pd.DataFrame(data=val, columns=col)
    else:
        poly_features = poly.fit_transform(data['X'].values[:, 1:])[:, 1:] # Remove first column (bias)
        col2 = []
        for i in np.arange(len(col1)-1, np.shape(poly_features)[1], 1):
            col2.append('f'+str(i+1))
        col = list(np.hstack([col1, col2]))
        val = np.hstack([values[:, 0][:, np.newaxis], poly_features])
        data['X'] = pd.DataFrame(data=val, columns=col)
    return data


def scale_data(data):
    X = data['X'].values[:, 1:]
    scale = MinMaxScaler(feature_range=(0, 1), copy=True)
    scale.fit(data['X'].values[:, 1:])
    X = scale.transform(X)
    data['X'].loc[:, data['X'].columns[1:]] = X
    return data


def linear_regression_model(train, validation, alpha, depth=None):

    X_train = train['X'].values[:, 1:]
    y_train = np.ravel(train['y'].values)
    X_validation = validation['X'].values[:, 1:]
    y_validation = np.ravel(validation['y'].values)

    models = {'type': ['ridge', 'decision tree', 'random forest'],
              'model': [Ridge(alpha=0.1, fit_intercept=fit_intercept, normalize=normalize, max_iter=max_iter, tol=tol,
                              random_state=random_state),
                        dt(criterion='mse', splitter='best', max_depth=depth, min_samples_split=2, min_samples_leaf=1,
                           random_state=random_state),
                        rfr(n_estimators=100, criterion='mse', max_depth=depth, min_samples_split=2, min_samples_leaf=1,
                            random_state=random_state)],
              'score_train': [], 'score_valid': [], 'mse_train': [], 'mse_valid': []}

    score_train = []
    score_valid = []
    mse_train = []
    mse_valid = []
    y_train_predict = []
    y_valid_predict = []
    for i in np.arange(0, len(models['type']), 1):
        m = models['model'][i]
        m.alpha = alpha
        m.fit(X_train, y_train)
        score_train.append(m.score(X_train, y_train))
        score_valid.append(m.score(X_validation, y_validation))
        y_train_predict.append(m.predict(X_train))
        y_valid_predict.append(m.predict(X_validation))
        mse_train.append(mse(y_train, y_train_predict[i]))
        mse_valid.append(mse(y_validation, y_valid_predict[i]))
    models['score_train'].append(score_train)
    models['score_valid'].append(score_valid)
    models['mse_train'].append(mse_train)
    models['mse_valid'].append(mse_valid)
    return models















