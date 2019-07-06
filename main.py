from utils import *


file_name = 'proj1_data.json'
data, feature_names, target_name = load_data(file_name)
train, validation, test = split_data(data, type='given', validation_size=.2, test_size=0.2, random_state=0)
word_count(train, num_feature=300)
# print(train)