from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import sys
import pandas as pd
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

def decision_tree_classifier(X, y, split_amount, X_test=None, y_test=None):
	'''
	X - inputs
	y- targets
	split_amount - percentage of dataset to split into test set
	'''
	train_split = 1 - split_amount
	X_train = None
	y_train = None
	if X_test is None and y_test is None:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_amount,
			train_size=train_split, shuffle=True)
	else:
		X_train = X
		y_train = y

	decision_tree = tree.DecisionTreeClassifier(criterion='gini')
	max_depth = range(2, 50)
	mean_cross_val_scores = []
	test_error = []
	test_accuracy = []

	train_scores, test_scores = validation_curve(decision_tree,
		X_train, y_train, param_name='max_depth'
		, param_range=max_depth, cv=5, n_jobs=1)

	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	best_depth = list(test_scores_mean).index(max(test_scores_mean))



	decision_tree.set_params(max_depth=best_depth+2)

	training_sizes = np.linspace(.1, 1.0, 5)
	train_sizes, train_scores_learn, test_scores_learn = learning_curve(decision_tree,
		X_train, y_train, train_sizes=training_sizes, cv=5, n_jobs=1)
	train_scores_mean_learn = np.mean(train_scores_learn, axis=1)
	train_scores_std_learn = np.std(train_scores_learn, axis=1)
	test_scores_mean_learn = np.mean(test_scores_learn, axis=1)
	test_scores_std_learn = np.std(test_scores_learn, axis=1)

	decision_tree.fit(X_train, y_train)
	measure_performance(X_test, y_test, decision_tree)


	lw = 2
	plt.figure()
	plt.grid()
	plt.title("Decision Tree Validation Curve")
	plt.plot(max_depth, train_scores_mean, label='training_score', color='darkorange')
	plt.fill_between(max_depth, train_scores_mean - train_scores_std,
	             train_scores_mean + train_scores_std, alpha=0.2,
	             color="darkorange", lw=lw)
	plt.plot(max_depth, test_scores_mean, label='cross_validation_score', color='navy')
	plt.fill_between(max_depth, test_scores_mean - test_scores_std,
	             test_scores_mean + test_scores_std, alpha=0.2,
	             color="navy", lw=lw)
	plt.legend()
	plt.xlabel('Max Depth')
	plt.ylabel('Score')



	title = "Decision Tree Learning Curve (Max Depth = " + str(best_depth + 2) + " )"
	plt.figure(2)
	plt.grid()
	plt.title(title)
	plt.fill_between(train_sizes, train_scores_mean_learn - train_scores_std_learn,
	                 train_scores_mean_learn + train_scores_std_learn, alpha=0.1,
	                 color="r")
	plt.fill_between(train_sizes, test_scores_mean_learn - test_scores_std_learn,
	                 test_scores_mean_learn + test_scores_std_learn, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean_learn, 'o-', color="r",
	         label="Training score")
	plt.plot(train_sizes, test_scores_mean_learn, 'o-', color="g",
	         label="Test score")
	plt.xlabel('Training Sizes')
	plt.ylabel('Score')
	plt.legend()
	plt.show()
	return decision_tree

def measure_performance(X, y, decision_tree):
	y_predict = decision_tree.predict(X)
	print("\n\nTest Accuracy: " + str(metrics.accuracy_score(y, y_predict)) + "\n\n")

	print("\n\nClassification report:\n\n")
	print(metrics.classification_report(y, y_predict))


	print("\n\nConfusion Matrix:\n\n")
	print(metrics.confusion_matrix(y, y_predict))


def export_tree_image(tree, feature_names):
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

def get_csv_data(filename, index=0):
	if os.path.exists(filename):
		print(filename + " found ")
		data_frame = pd.read_csv(filename, index_col=index)
		return data_frame
	else:
		print("file not found.")

def create_inputs(dataframe, target):
	features = list(dataframe.columns[:11])
	y = dataframe[target]
	X = dataframe[features]
	print(np.mean(y.values))

	return X.values, y.values

def encode_features(features, dataframe, reduce_classes=True):
	'''
	encodes features that are string values to integers
	features- features to encode
	dataframe - full pandas data table
	'''
	dataframe_copy = dataframe.copy()
	le = LabelEncoder()

	for feature in features:
		dataframe_copy[feature] = le.fit_transform(dataframe_copy[feature])

	if reduce_classes:
		for i in range(0, len(dataframe_copy["G3"].values)):
			if dataframe_copy["G3"].values[i] >= 0 and dataframe_copy["G3"].values[i] <=4:
				dataframe_copy["G3"].values[i] = 1
			elif dataframe_copy["G3"].values[i] >= 5 and dataframe_copy["G3"].values[i] <=10:
				dataframe_copy["G3"].values[i] = 2
			elif dataframe_copy["G3"].values[i] >= 11 and dataframe_copy["G3"].values[i] <= 15:
				dataframe_copy["G3"].values[i] = 3
			elif dataframe_copy["G3"].values[i] >= 16 and dataframe_copy["G3"].values[i] <= 20:
				dataframe_copy["G3"].values[i] = 4
	return dataframe_copy



def combine_csv(file1, file2):
	'''
	Combines two csv files with same column info into one dataframe
	'''
	data_frame1 = []
	data_frame2 = []
	if os.path.exists(file1):
		print(file1 + " found ")
		data_frame1 = pd.read_csv(file1, index_col=0)
	else:
		print("file not found.")

	if os.path.exists(file2):
		print(file2 + " found ")
		data_frame2 = pd.read_csv(file2, index_col=0)
	else:
		print("file not found.")

	full_data = pd.concat([data_frame1, data_frame2], ignore_index=True)
	return full_data

def show_distribution(y):
	'''
	y - target values to show distribution of
	'''
	plt.figure()
	plt.hist(y)
	plt.title("Wine Sample Distribution")
	plt.xlabel("Wine Quality Scores")
	plt.ylabel("Frequency")
	plt.show()


#df = get_csv_data("student-mat.csv")

if sys.argv[1] == 'student':
	df = combine_csv("student-mat.csv", "student-por.csv")
	print(len(df))
	all_classes = list(df.columns)
	df_string = list(df.select_dtypes(include=['object']).columns)
	df_encoded = encode_features(df_string, df, reduce_classes=False)
	X,y = create_inputs(df_encoded, "G3")
	show_distribution(y)
	dt = decision_tree_classifier(X, y, .3)

	#running decision trees but with 4 classes instead of 20
	df_encoded = encode_features(df_string, df)
	X,y = create_inputs(df_encoded, "G3")
	dt = decision_tree_classifier(X, y, .3)


elif sys.argv[1] == 'exoplanet':
	dfexotrain = get_csv_data("exoTrain.csv", index=None)
	dfexotest = get_csv_data("exoTest.csv", index=None)
	X_train,y_train = create_inputs(dfexotrain, "LABEL")
	X_test, y_test = create_inputs(dfexotest, "LABEL")
	dt2 = decision_tree_classifier(X_train, y_train, 0.0, X_test=X_test, y_test=y_test)

elif sys.argv[1] == 'wine':
	dfwine = get_csv_data("winequality_white.csv", index=None)
	X,y = create_inputs(dfwine, "quality")
	show_distribution(y)
	split_amount = 0.3
	dt3 = decision_tree_classifier(X,y,split_amount)





#export_tree_image(dt, features)