from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import PlaintextCorpusReader as CorpusRead
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk
import re
from collections import Counter
import random
import pickle
import numpy as np

PATH_DATA = "./data/train.csv"
PATH_OBJ = "./objects/"
FEATURE_FILE = PATH_OBJ + "features/features_{}.pkl"
PATH_TRAINING_DATA = "./training_data/"

TRANING_FILE = PATH_TRAINING_DATA + "train_{}.pkl"

CLASSIFIER_FILE = PATH_OBJ + "classifier/NBclassifier_{}.pkl"

word_regex = r'[a-z]+'
stop_words = set(stopwords.words('english'))

features = []



def create_features(train_df, N_MOST_COMMON):
	# features = []
	print("Length of training set : ", len(train_df))
	all_words = train_df["Description"].tolist()

	all_words = word_tokenize(" ".join(all_words))

	all_words = Counter(all_words)
	common_words = all_words.most_common(N_MOST_COMMON)
	for word, _ in common_words:
		word = word.lower()
		# re.search(word_regex, word) and 
		if word not in stop_words:
			features.append(word)

	# return features

# def find_features(desc):
# 		feature_set = {}
# 		for feature in features:
# 			if feature in desc:
# 				feature_set[feature] = True
# 			else:
# 				feature_set[feature] = False
# 		return feature_set

def find_features_matrix(desc):
		feature_set = []
		i = 0
		for feature in features:
			if feature in desc:
				feature_set.append(1)
			else:
				feature_set.append(0)
			i += 1
		return feature_set


def save_to_pickle(file_name,object_to_save):
	with open(file_name, 'wb') as f:
		pickle.dump(object_to_save, f)


def load_from_pickle(file_name):
	try:
		with open(file_name, 'rb') as f:
			data = pickle.load(f)
		return False, data
	except:
		return True, None

def get_training_data(train_df):
	training_data = []
	y_label = []
	for idx, row in train_df.iterrows():
		desc = word_tokenize(row["Description"].lower())
		y_label.append(row["Is_Response"])
		training_data.append(find_features_matrix(desc))
	random.shuffle(training_data)
	return training_data, y_label

def main():
	global features
	print("Reading data .....")
	train_df = pd.read_csv(PATH_DATA)
	classes = train_df["Is_Response"].unique().tolist()
	# for i in range(4,16):
	VERSION = 2
	N_MOST_COMMON = VERSION * 1000
	print("Creating features ....., for ",N_MOST_COMMON," most common words...")
	failure , features = load_from_pickle(FEATURE_FILE.format(N_MOST_COMMON))
	if failure:
		# features = create_features(train_df)
		features = []
		create_features(train_df,N_MOST_COMMON)
		save_to_pickle(FEATURE_FILE.format(N_MOST_COMMON), features)
	
	print("Total features : ", len(features))

	number_of_batches = 5
	batch_size = int(len(train_df) / number_of_batches)

	train_df = train_df.sample(frac=1).reset_index(drop=True)
	NBclassifier = MLPClassifier(hidden_layer_sizes=(1000,1000,1000),activation='logistic',learning_rate_init=0.005,max_iter=200, verbose=True)#MultinomialNB()

	print("Number of classes : ", classes)

	print("Training {} batches of data ......".format(number_of_batches))
	for i in range(number_of_batches):
		start = i*batch_size
		end = (i+1)*batch_size

		if i == (number_of_batches - 1):
			print("Preparing test set :", "start : ", start)
			batch_df = train_df[start:]
			data_set, y_label = get_training_data(batch_df)
			print("Testing for accuracy .....")
			print("Accuracy : ",NBclassifier.score(data_set, y_label))
		else:
			batch_df = train_df[start:end]
			data_set, y_label = get_training_data(batch_df)
			print("Training data for batch {} .....".format(i))
			if i == 0:
				NBclassifier.partial_fit(data_set, y_label, classes=classes)
			else:
				NBclassifier.partial_fit(data_set, y_label)
		


	# failure, NBclassifier = load_from_pickle(CLASSIFIER_FILE.format(N_MOST_COMMON))
	# training_failure, training_data = load_from_pickle(TRANING_FILE.format(N_MOST_COMMON))

	# if failure: # or training_failure:
		
		# if training_failure:	
		# print("Preparing training data .....")
		# training_data = get_training_data(train_df)
			# save_to_pickle(TRANING_FILE.format(N_MOST_COMMON), training_data)
		# else:
		# 	print("Loaded training data successfully, No of examples : ", len(training_data))	
		# DIV = int(len(training_data)*0.8)
		# train, test = training_data[:DIV], training_data[DIV:]
		# print("Training classifier .....")
		# NBclassifier = nltk.NaiveBayesClassifier.train(train)
		# save_to_pickle(CLASSIFIER_FILE.format(N_MOST_COMMON), NBclassifier)
	# else:
	# 	# DIV = int(len(training_data)*0.8)
	# 	train, test = training_data[:DIV], training_data[DIV:]
		# print("Accuracy : ",nltk.classify.accuracy(NBclassifier, test))

if __name__ == '__main__':
	main()


