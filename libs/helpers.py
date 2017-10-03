from config.Config import Config
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


def save(fileName, objectToSave):
	with open(fileName, 'wb') as handle:
		pickle.dump(objectToSave, handle)

def load(fileName):
	with open(fileName, 'rb') as handle:
		loadedObject = pickle.load(handle)
	return loadedObject


def read_csv_to_df(fileName):
	fileToRead = Config.DataPath + fileName
	data = pd.read_csv(fileToRead)
	return data


def get_train_test_set_df(data, test_size=0.25):
	train, test = train_test_split(data, test_size=test_size)
	return train, test


def seperate_positive_negative_case_df(data,key, pos_val, neg_val):
	pos = data[data[key] == pos_val]
	neg = data[data[key] == neg_val]
	return pos, neg



def get_training_and_testing_df():
	# reads csv and loads it in memory as dataframe
	data_set = read_csv_to_df(Config.TrainingDataFile)

	# splits dataframe of data set into train and test set
	train_set, test_set = get_train_test_set_df(data_set)

	print("Length of training set : {0} and testing set {1}".format(len(train_set), len(test_set)))

	return train_set, test_set

def get_training_data(train_set):
	
	reviews_train = train_set['Description'].tolist()
	labels_train = train_set['Is_Response'].tolist()

	return reviews_train, labels_train

def get_pos_neg(data_set):
	pos, neg = seperate_positive_negative_case_df(data_set, 'Is_Response', 'happy', 'not happy')
	return pos['Description'].tolist(), neg['Description'].tolist()
