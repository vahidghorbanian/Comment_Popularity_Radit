import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error as mse
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
    type = data['type']
    txt = data['X']['text'].values
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
    return feature_name, feature_count


def add_total_word_count(data):
    type = data['type']
    text = data['X']['text'].values
    print('Calculating text features counts in ' + type + ' set ...')
    count = []
    for i in np.arange(0, len(text), 1):
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

def linear_regression_model(train, validation, alpha):
    # train['X'].drop(['controversiality'], axis=1)
    # validation['X'].drop(['controversiality'], axis=1)

    X_train = train['X'].values[:, 1:]
    y_train = np.ravel(train['y'].values)
    X_validation = validation['X'].values[:, 1:]
    y_validation = np.ravel(validation['y'].values)

    models = {'type': ['ridge'],
              'model': [Ridge(alpha=0.1, fit_intercept=fit_intercept, normalize=normalize, max_iter=max_iter, tol=tol,
                              random_state=random_state)],
              'coef':[], 'intercept': [], 'score_train': [], 'score_valid': [], 'mse_train': [], 'mse_valid': [],
              'y_train_predict': [], 'y_valid_predict': [], 'alpha': []}

    for i in np.arange(0, len(models['type']), 1):
        m = models['model'][i]
        coef = []
        intercept = []
        score_train = []
        score_valid = []
        mse_train = []
        mse_valid = []
        y_train_predict = []
        y_valid_predict = []
        for j in np.arange(0, len(alpha), 1):
            m.alpha = alpha[j]
            m.fit(X_train, y_train)
            coef.append(m.coef_)
            intercept.append(m.intercept_)
            score_train.append(m.score(X_train, y_train))
            score_valid.append(m.score(X_validation, y_validation))
            y_train_predict.append(m.predict(X_train))
            y_valid_predict.append(m.predict(X_validation))
            mse_train.append(mse(y_train, y_train_predict[j]))
            mse_valid.append(mse(y_validation, y_valid_predict[j]))
        models['coef'].append(coef)
        models['intercept'].append(intercept)
        models['score_train'].append(score_train)
        models['score_valid'].append(score_valid)
        models['mse_train'].append(mse_train)
        models['mse_valid'].append(mse_valid)
        models['y_train_predict'].append(y_train_predict)
        models['y_valid_predict'].append(y_valid_predict)
    models['alpha'] = alpha
    return models


def plot_results(models, train, validation):
    y_train = np.ravel(train['y'].values)
    y_validation = np.ravel(validation['y'].values)

    for i in np.arange(0, len(models['type']), 1):
        plt.figure(figsize=(5, 3))
        plt.bar(models['alpha'], models['score_train'][i], align='center', alpha=0.7)
        plt.xlabel('alpha')
        plt.ylabel('training score')
        plt.title(models['type'][i])
        plt.tight_layout()

        plt.figure(figsize=(5, 3))
        plt.bar(models['alpha'], models['score_valid'][i], align='center', alpha=0.7)
        plt.xlabel('alpha')
        plt.ylabel('validation score')
        plt.title(models['type'][i])
        plt.tight_layout()

        plt.figure(figsize=(5, 3))
        plt.bar(models['alpha'], models['mse_train'][i], align='center', alpha=0.7)
        plt.xlabel('alpha')
        plt.ylabel('MSE training')
        plt.title(models['type'][i])
        plt.tight_layout()

        plt.figure(figsize=(5, 3))
        plt.bar(models['alpha'], models['mse_valid'][i], align='center', alpha=0.7)
        plt.xlabel('alpha')
        plt.ylabel('MSE validation')
        plt.title(models['type'][i])
        plt.tight_layout()

        plt.figure(figsize=(5, 3))
        plt.scatter(y_train, models['y_train_predict'][i], alpha=0.7)
        plt.xlabel('target')
        plt.ylabel('prediction')
        plt.title(models['type'][i])
        plt.tight_layout()

    return 0
















