from libs.helpers import read_csv_to_df, get_train_test_set_df, seperate_positive_negative_case_df, get_training_and_testing_df, get_training_data, get_pos_neg, save
from config.Config import Config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline


def save_classfier(name, classifier):
	classifierFileName = Config.ModelPath + name + "_v" + Config.VersionCode + ".pickle"
	print("Saving classifier ...")
	save(classifierFileName, classifier)

def train():
	# fetching training data
	train_set, test_set = get_training_and_testing_df(clean=True)
	reviews_train, labels_train = get_training_data(train_set)

	pos, neg = get_pos_neg(test_set)
	pos_label = ['happy'] * len(pos)
	neg_label = ['not happy'] * len(neg)

	print("Test Cases :\n\tNo. of positive examples : {0}, No. of negative examples : {1}".format(len(pos), len(neg)))

	# Tf-Idf Vectors for features
	vectorizer = TfidfVectorizer(max_df=0.5, stop_words='english')

	X_train = vectorizer.fit_transform(reviews_train)
	X_pos = vectorizer.transform(pos)
	X_neg = vectorizer.transform(neg)
	vectorizerFileName = Config.VectorizerPath + "TfidfVectorizer_v" + Config.VersionCode + ".pickle"

	print("Saving vectorizer ...")
	save(vectorizerFileName, vectorizer)

	# deleting unnecessary variables
	del train_set, test_set, pos, neg, reviews_train

	# feature selection 
	print("Selecting {0} features".format(Config.NumberOfFeatures))
	ch2 = SelectKBest(chi2, k=Config.NumberOfFeatures)
	X_train = ch2.fit_transform(X_train, labels_train)
	X_pos = ch2.transform(X_pos)
	X_neg = ch2.transform(X_neg)
	featureSelectorFileName = Config.FeaturePath + "ch2_v" + Config.VersionCode + ".pickle"

	print("Saving feature selector ...")
	save(featureSelectorFileName, ch2)


	X_train = X_train.todense()
	X_pos = X_pos.todense()
	X_neg = X_neg.todense()

	classifierList = [('LinearSVC', LinearSVC()), ('Perceptron',Perceptron()), ('SGDClassifier',SGDClassifier())]

	for name, classifier in classifierList:
		print("Training {0} classifier ...".format(name))
		classifier.fit(X_train, labels_train)

		print("Testing for accuracy ...")
		print("Acuracy for {0} of positive cases : {1}".format(name,classifier.score(X_pos, pos_label)))
		print("Acuracy for {0} of negative cases : {1}".format(name,classifier.score(X_neg, neg_label)))

		save_classfier(name, classifier)


	voteClassifier = VotingClassifier(estimators=classifierList)
	name = 'voteClassifier'

	print("Training {0} classifier ...".format(name))
	voteClassifier.fit(X_train, labels_train)
	print("Testing for accuracy ...")
	print("Acuracy for {0} of positive cases : {1}".format(name,voteClassifier.score(X_pos, pos_label)))
	print("Acuracy for {0} of negative cases : {1}".format(name,voteClassifier.score(X_neg, neg_label)))
	save_classfier(name, voteClassifier)


if __name__ == '__main__':
	train()