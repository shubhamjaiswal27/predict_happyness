from libs.helpers import read_csv_to_df, get_train_test_set_df, seperate_positive_negative_case_df
from config.Config import Config
from sklearn.feature_extraction.text import TfidfVectorizer



def get_training_data():
	# reads csv and loads it in memory as dataframe
	data_set = read_csv_to_df(Config.TrainingDataFile)

	# splits dataframe of data set into train and test set
	train_set, test_set = get_train_test_set_df(data_set)

	print("Length of training set : {0} and testing set {1}".format(len(train_set), len(test_set)))

	reviews_train = train_set['Description'].tolist()
	labels_train = train_set['Is_Response'].tolist()

	reviews_test = test_set['Description'].tolist()
	labels_test = test_set['Is_Response'].tolist()

	return reviews_train, labels_train, reviews_test, labels_test


def main():
	# fetching training data
	reviews_train, labels_train, reviews_test, labels_test = get_training_data()

	# Tf-Idf Vectors for features
	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

	X_train = vectorizer.fit_transform(reviews_train)
	X_test = vectorizer.fit_transform(reviews_test)

	feature_names = vectorizer.get_feature_names()

	print("Features length : {0}".format(len(feature_names)))

if __name__ == '__main__':
	main()