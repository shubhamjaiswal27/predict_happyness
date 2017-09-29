from config.Config import Config
import pandas as pd
from sklearn.model_selection import train_test_split


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

