import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import multiprocessing


# Load data
def load_data(file_name):
    file = open(file_name)
    data = pd.DataFrame(data=json.load(file))
    data = data[['text', 'children', 'controversiality', 'is_root', 'popularity_score']]
    data['is_root'] = 1*data['is_root']
    num_samples = len(data)
    features = list(data.columns[:-1])
    target = data.columns[-1]
    print('number of samples: ', num_samples)
    print('features name:', features)
    print('target name:', target)
    return data, features, target


# split data
def split_data(data, type='given', validation_size=.2, test_size=0.1, random_state=0):
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
    return {'type': 'train', 'X': X_train, 'y': y_train}, \
           {'type': 'validation', 'X': X_validation, 'y': y_validation},\
           {'type': 'test', 'X': X_test, 'y': y_test}


def word_count(data, num_feature=160):
    X = data['X']['text'].values
    for i in np.arange(0, len(X), 1):
        X[i] = X[i].lower()
    words = X.sum().split()
    words_unique = np.unique(words)
    print('\ntotal number of words in database: ', len(words))
    print('total number of unique words in database: ', len(words_unique))
    count = np.zeros((len(words_unique), 1))
    for i in np.arange(0, len(words_unique), 1):
        count[i] = words.count(words_unique[i])
        print(i)
    features = []
    features_count = []
    count = list(count)
    words_unique = list(words_unique)
    for i in np.arange(0, num_feature, 1):
        idx = np.argmax(count)
        features.append(words_unique[idx])
        features_count.append(count[idx])
        count.pop(idx)
        words_unique.pop(idx)
    features = np.vstack(features)
    features_count = np.vstack(features_count)

    return features, features_count













