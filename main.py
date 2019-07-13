from utils import *
import matplotlib.pyplot as plt
import time


# Initialization
num_text_feature = 160
order_poly_feature = 1
include_bias = False
add_word_count = False
add_text_features = True
add_poly_features = True
alpha = 0

# Run
t = time.time()

file_name = 'proj1_data.json'
data, main_feature_name, target_name = load_data(file_name)
train, validation, test = split_data(data, type='given', validation_size=.2, test_size=0.01)

high_rank_feature_name, high_rank_feature_count, num_total_words = high_rank_features(data.values[:, :-1],
                                                                                      num_feature=num_text_feature)

if add_word_count == True:
    train = add_total_word_count(train, num_total_words)
    validation = add_total_word_count(validation, num_total_words)
    test = add_total_word_count(test, num_total_words)

if add_text_features == True:
    train = add_txt_features(train, high_rank_feature_name)
    validation = add_txt_features(validation, high_rank_feature_name)
    test = add_txt_features(test, high_rank_feature_name)

if add_poly_features == True:
    if order_poly_feature >= 2:
        train = add_interaction_terms(train, order=order_poly_feature, include_bias=include_bias)
        validation = add_interaction_terms(validation, order=order_poly_feature, include_bias=include_bias)
        test = add_interaction_terms(test, order=order_poly_feature, include_bias=include_bias)

print('training set shape:', train['X'].shape)
print('validation set shape:', validation['X'].shape)
print('test set shape:', test['X'].shape)

models = linear_regression_model(train, validation, alpha, depth=20)

elapsed = time.time() - t
print('\nRun time: ', elapsed, 's')

plt.bar(models['type'], models['score_train'])
plt.ylabel('R2: Training')
plt.figure()
plt.bar(models['type'], models['score_valid'])
plt.ylabel('R2: Validation')
plt.figure()
plt.bar(models['type'], models['mse_train'])
plt.ylabel('MSE: Training')
plt.figure()
plt.bar(models['type'], models['mse_valid'])
plt.ylabel('MSE: Validation')
plt.show()