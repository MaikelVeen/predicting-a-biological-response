import pandas as pd 
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os 
import math 

def bprint(text):
	""" Prints the text in blue """
	print(f'\033[94m {text} \033[0m')


def gprint(text):
	""" Prints the text in green """
	print(f'\033[92m {text} \033[0m')


def cprint(text):
	""" Prints the text in green """
	print(f'\033[36m {text} \033[0m')


def save_submission_csv(predictions, method_name):
	"""Saves the weights and final prediction to the file sytem"""
	date = datetime.now()

	# Create data dictionary 
	data = {'MoleculeId': np.arange(
			1, len(predictions) + 1), 'PredictedProbability': predictions[:, 1]}

	# Create dataframa from dictionary and save to csv
	df = pd.DataFrame(data, columns=['MoleculeId', 'PredictedProbability'])

	# Construct OS indepent path
	dir_path = os.path.dirname(os.path.abspath(__file__))
	filename = f'{dir_path}/submissions/submission-{date.strftime("%H_%M_%S")}-{method_name}.csv'

	# Save to csv file
	df.to_csv(filename, index=False, header=True)

def load_data(pre_process=False):
	""" Loads and returns a data set"""
	dir_path = os.path.dirname(os.path.realpath(__file__))

	training_data = pd.read_csv(f'{dir_path}/data/train.csv')
	test_data = pd.read_csv(f'{dir_path}/data/test.csv')
	
	training_data = training_data.drop(['D353', 'D1502', 'D1676', 'D399', 'D428', 'D908', 'D1639', 'D468', 'D801', 'D544', 'D400', 'D771', 'D371', 'D833', 'D1683', 'D484', 'D910', 'D404', 'D465', 'D500', 'D708', 'D896', 'D823', 'D415', 'D682', 'D615', 'D467', 'D898', 'D849', 'D632', 'D605', 'D345', 'D797', 'D1529', 'D1542', 'D1296', 'D631', 'D235', 'D284', 'D876', 'D630', 'D616', 'D619', 'D570', 'D450', 'D339', 'D680', 'D874', 'D581', 'D1240'], axis=1)
	test_data = test_data.drop(['D353', 'D1502', 'D1676', 'D399', 'D428', 'D908', 'D1639', 'D468', 'D801', 'D544', 'D400', 'D771', 'D371', 'D833', 'D1683', 'D484', 'D910', 'D404', 'D465', 'D500', 'D708', 'D896', 'D823', 'D415', 'D682', 'D615', 'D467', 'D898', 'D849', 'D632', 'D605', 'D345', 'D797', 'D1529', 'D1542', 'D1296', 'D631', 'D235', 'D284', 'D876', 'D630', 'D616', 'D619', 'D570', 'D450', 'D339', 'D680', 'D874', 'D581', 'D1240'], axis=1)
	
	training_X = _get_X(training_data)
	training_y = _get_y(training_data)
	test_X = _get_X(test_data, False)

	if pre_process:
		# Preprocess the ddata 
		merged_data = np.concatenate((training_X, test_X))
		pca_data = StandardScaler().fit_transform(merged_data)

		# Create PCA model, we only elimate the variables that have
		# no effect on the variance
		pca = PCA(1646)

		# Fit and transform the set using the model
		pca.fit(pca_data)
		training_X  = pca.transform(training_X)
		test_X = pca.transform(test_X)
		return training_X, training_y, test_X

	return training_X, training_y, test_X


def _get_X(data, training=True):
	if training:
		X = data.iloc[:, 1:]
		return X.to_numpy()
	else:
		return data.to_numpy()


def _get_y(data):
	y = data.iloc[:, 0]
	return y.to_numpy()
