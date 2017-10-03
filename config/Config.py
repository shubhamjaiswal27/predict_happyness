class Config:

	DataPath = "data/"
	ModelPath = "models/"
	FeaturePath = "feature_selector/"
	VectorizerPath = "vectorizer/"
	SubmissionPath = "output/"

	VersionCode = "1"

	TrainingDataFile = "train.csv"
	TestingDataFile = "test.csv"

	NumberOfFeatures = 1500

	CleanData = True

class Clasify:
	ModelFileName = Config.ModelPath + "LinearSVC_v" + Config.VersionCode + ".pickle"
	FeatureFileName = Config.FeaturePath + "ch2_v" + Config.VersionCode + ".pickle"
	VectorizerFileName = Config.VectorizerPath + "TfidfVectorizer_v" + Config.VersionCode + ".pickle"
	