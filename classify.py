from libs.helpers import read_csv_to_df, cleanData, load
from config.Config import Config, Clasify


def classify():
	test = read_csv_to_df(Config.TestingDataFile)

	# steps for cleaning data if config is set to true
	if Config.CleanData:
		print("cleaning data ...")
		test['Description'] = test['Description'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True))

	reviews = test['Description'].tolist()

	# load all models
	print("loading all saved models ...")
	vectorizer = load(Clasify.VectorizerFileName)
	ch2 = load(Clasify.FeatureFileName)
	classfier = load(Clasify.ModelFileName)

	print("computing features ...")
	X_test = vectorizer.transform(reviews)
	X_test = ch2.transform(X_test)

	print("perdicting ...")
	Is_Response = classfier.predict(X_test)
	test['Is_Response'] = Is_Response

	print("saving final output ...")
	test = test[['User_ID', 'Is_Response']]
	submissionFile = Config.SubmissionPath + "output_v" + Config.VersionCode + ".csv"
	test.to_csv(submissionFile, index=False)

if __name__ == '__main__':
	classify()