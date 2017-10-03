# predict_happyness


For predicting whether the user is happy or not based on review, I have used a voting classifier which internally uses three classifier.

1) LinearSVC
2) Perceptron
3) SGDClassifier

For feature extraction i have used TfidfVectorizer and for feature selection i have used SelectKBest from scikit learn.
Feature are selected by using chi2 in scikit learn.