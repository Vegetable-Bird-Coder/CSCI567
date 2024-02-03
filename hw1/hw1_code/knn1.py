import numpy as np
import json
import collections
import matplotlib.pyplot as plt


def data_processing(data):
	train_set, valid_set, test_set = data['train_data'], data['val_data'], data['test_data']
	Xtrain = train_set["features"]
	ytrain = train_set["labels"]
	Xval = valid_set["features"]
	yval = valid_set["labels"]
	Xtest = test_set["features"]
	ytest = test_set["labels"]

	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)

	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)
	
	return Xtrain, ytrain, Xval, yval, Xtest, ytest


def data_processing_with_transformation(data, do_minmax_scaling=True, do_normalization=False):
	train_set, valid_set, test_set = data['train_data'], data['val_data'], data['test_data']
	Xtrain = train_set["features"]
	ytrain = train_set["labels"]
	Xval = valid_set["features"]
	yval = valid_set["labels"]
	Xtest = test_set["features"]
	ytest = test_set["labels"]

	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)

	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)
	
	# We load data from json here and turn the data into numpy array
	# You can further perform data transformation on Xtrain, Xval, Xtest

	# Min-Max scaling
	def minmax_scaling(X):
		max_features_cols = np.max(X, axis=0)
		max_features_cols = max_features_cols[np.newaxis, :]
		min_features_cols = np.min(X, axis=0)
		min_features_cols = min_features_cols[np.newaxis, :]
		return (X - min_features_cols) / (max_features_cols - min_features_cols)		
	if do_minmax_scaling:
		
		#####################################################
		#				 YOUR CODE HERE					    #
		#####################################################
		Xtrain = minmax_scaling(Xtrain)
		Xval = minmax_scaling(Xval)
		Xtest = minmax_scaling(Xtest)

	# Normalization
	def normalization(x):
		#####################################################
		#				 YOUR CODE HERE					    #
		#####################################################
		epsilon = 1e-10
		featureLen = np.sqrt(np.sum(x ** 2,axis=0)) + epsilon
		return x / featureLen[np.newaxis, :]
	
	if do_normalization:
		Xtrain = normalization(Xtrain)
		Xval = normalization(Xval)
		Xtest = normalization(Xtest)

	return Xtrain, ytrain, Xval, yval, Xtest, ytest


def compute_l2_distances(Xtrain, X):
	"""
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Euclidean distance between the ith test point and the jth training
	  point.
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	num_train = Xtrain.shape[0];
	num_test = X.shape[0];
	dists = np.empty((num_test, num_train))
	for i in range(num_test):
		for j in range(num_train):
			dists[i][j] = np.sum((Xtrain[j] - X[i])**2)
			dists[i][j] = np.sqrt(dists[i][j])
	# print(Xtrain[j])
	# print(X[i])
	# print(dists[1][1])
	return dists


def compute_cosine_distances(Xtrain, X):
	"""
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Cosine distance between the ith test point and the jth training
	  point.
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	num_train = Xtrain.shape[0];
	num_test = X.shape[0];
	dists = np.empty((num_test, num_train))
	for i in range(num_test):
		for j in range(num_train):
			normx = np.sqrt(np.sum(Xtrain[j] ** 2))
			normxprime = np.sqrt(np.sum(X[i] ** 2))
			if normx == 0 or normxprime == 0:
				dists[i][j] = 1
			else:
				dists[i][j] = 1 - ((Xtrain[j].T.dot(X[i])) / (normx * normxprime))
	return dists


def predict_labels(k, ytrain, dists):
	"""
	Given a matrix of distances between test points and training points,
	predict a label for each test point.
	Inputs:
	- k: The number of nearest neighbors used for prediction.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  gives the distance betwen the ith test point and the jth training point.
	Returns:
	- ypred: A numpy array of shape (num_test,) containing predicted labels for the
	  test data, where y[i] is the predicted label for the test point X[i].
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	# num_train = ytrain.shape[1]
	num_test = dists.shape[0]
	ypred = np.zeros(num_test)
	for i in range(dists.shape[0]):
		firstKSmallestIndex = np.argpartition(dists[i], k)[:k]
		numberOfOne = 0
		for index in firstKSmallestIndex:
			numberOfOne += ytrain[index]
		if numberOfOne > k / 2:
			ypred[i] = 1
	return ypred


def compute_error_rate(y, ypred):
	"""
	Compute the error rate of prediction based on the true labels.
	Inputs:
	- y: A numpy array with of shape (num_test,) where y[i] is the true label
	  of the ith test point.
	- ypred: A numpy array with of shape (num_test,) where ypred[i] is the
	  prediction of the ith test point.
	Returns:
	- err: The error rate of prediction (scalar).
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	errorNum = 0;
	for i in range(y.shape[0]):
		if y[i] != ypred[i]:
			errorNum += 1
	return errorNum / y.shape[0]


def find_best_k(K, ytrain, dists, yval):
	"""
	Find best k according to validation error rate.
	Inputs:
	- K: A list of ks.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the distance between the ith test point and the jth training
	  point.
	- yval: A numpy array with of shape (num_val,) where y[i] is the true label
	  of the ith validation point.
	Returns:
	- best_k: The k with the lowest error rate.
	- validation_error: A list of error rate of different ks in K.
	- best_err: The lowest error rate we get from all ks in K.
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	validation_error = []
	for k in K:
		ypred = predict_labels(k, ytrain, dists)
		err = compute_error_rate(yval, ypred)	
		validation_error.append(err)	
	best_err = min(validation_error)
	min_index = validation_error.index(best_err)
	best_k = K[min_index]
	return best_k, validation_error, best_err


def main():
	input_file = 'breast_cancer_dataset.json'
	output_file = 'knn_output.txt'

	#==================Problem Set 1.1=======================

	with open(input_file) as json_data:
		data = json.load(json_data)

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.1")
	print()

	#==================Problem Set 1.2=======================

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_with_transformation(data, do_minmax_scaling=False, do_normalization=True)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.2 when using normalization")
	print()

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_with_transformation(data, do_minmax_scaling=True, do_normalization=False)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.2 when using minmax_scaling")
	print()
	
	#==================Problem Set 1.3=======================

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)
	dists = compute_cosine_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.3, which use cosine distance")
	print()

	#==================Problem Set 1.4=======================
	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)

	#======performance of different k in training set=====
	K = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################

	
	#==========select the best k by using validation set==============
	dists = compute_l2_distances(Xtrain, Xval)
	best_k, validation_error, best_err = find_best_k(K, ytrain, dists, yval)

	#===============test the performance with your best k=============
	dists = compute_l2_distances(Xtrain, Xtest)
	ypred = predict_labels(best_k, ytrain, dists)
	test_err = compute_error_rate(ytest, ypred)
	print("In Problem Set 1.4, we use the best k = ", best_k, "with the best validation error rate", best_err)
	print("Using the best k, the final test error rate is", test_err)
	#====================write your results to file===================
	f=open(output_file, 'w')
	for i in range(len(K)):
		f.write('%d %.3f' % (K[i], validation_error[i])+'\n')
	f.write('%s %.3f' % ('test', test_err))
	f.close()

	print("4.3.1")
	dists = compute_l2_distances(Xtrain, Xtrain)
	best_k_Xtrain, validation_error_Xtrain, best_err_Xtrain = find_best_k(K, ytrain, dists, ytrain)
	fig, ax = plt.subplots(figsize = (8,5))
	# ax = fig.add_subplot(2,1,1)
	ax.plot(K, validation_error_Xtrain, 'bo-', label='using training set')

	dists = compute_l2_distances(Xtrain, Xval)
	best_k_Xval, validation_error_Xval, best_err_Xval = find_best_k(K, ytrain, dists, yval)
	# ax = fig.add_subplot(2,1,2)
	ax.plot(K, validation_error_Xval, 'ro-', label='using validation set')
	ax.legend()

	dists = compute_l2_distances(Xtrain, Xtest)
	best_k_Xtest, validation_error_Xtest, best_err_Xtest = find_best_k(K, ytrain, dists, ytest)
	# ax = fig.add_subplot(2,1,2)
	# ax.plot(K, validation_error_Xtest, 'go-', label='using test set')
	# ax.legend()

	print("when k = 6, error rate on test data set is " + str(validation_error_Xtest[3]))
	# print("K = " + str(K))
	# print("error rates on test data set is " + str(validation_error_Xtest))
	plt.xticks(np.arange(min(K), max(K)+1, 1))
	plt.xlabel('K')
	plt.ylabel('error_rate')	
	plt.show()


if __name__ == "__main__":
	main()
