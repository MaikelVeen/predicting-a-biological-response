import helper
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from models.xgboost_model import XGBoostClassifier
from models.ann_model import ANNClassifier

class Ensemble():
	""" Wrapper around the main algorithm """

	def __init__(self, folds=10, verbose=True, estimators=1000):
		self.folds = folds
		self.verbose = verbose
		self.estimators = estimators
		self._load_data()

	def _load_data(self):
		""" Loads the data using the helper """

		if self.verbose:
			helper.bprint("Loading data from csv.")

		self.x, self.y, self.submission_data = helper.load_data()

	def run(self):
		""" Main algorithmic loop"""
		np.random.seed(420)

		folds = self._get_folds()
		classifiers, classifier_count = self._build_classifiers(0, self.estimators)

		# Init train test split blend sets
		blend_train = np.zeros((self.x.shape[0], classifier_count))
		blend_test = np.zeros((self.submission_data.shape[0], classifier_count))

		# Loop through the classifiers
		for c_index, classifier in enumerate(classifiers):
			if self.verbose:
				helper.bprint(f"Classifier: {c_index} - {classifier}")

			fold_sum = np.zeros((self.submission_data.shape[0], len(folds)))

			# Loop trough all the k-folds
			for f_index, (train, test) in enumerate(folds):
				helper.cprint(f"Training fold {f_index}")

				x_train, x_test, y_train, y_test = self._get_sets(train, test)
				classifier.fit(x_train, y_train)

				# Predict on test split set
				test_pred = np.array(classifier.predict_proba(x_test))
				blend_train[test, c_index] = test_pred[:, 1]

				# Predit on submission data
				sub_pred = np.array(classifier.predict_proba(self.submission_data))
				fold_sum[:, f_index] = sub_pred[:, 1]

			blend_test[:, c_index] = fold_sum.mean(1)
			helper.gprint(f"Done training {classifier}")

		# Blend the classifiers
		final_sub = self._blend(blend_train, blend_test)

		# Save to csv after stretching
		self._save(self._stretch(final_sub))

	def _get_folds(self):
		""" Returns indices to split test and training data """

		if self.verbose:
			helper.bprint("Initializing Stratified K-Folds cross-validator")

		kfold_cv = StratifiedKFold(self.folds)
		return list(kfold_cv.split(self.x, self.y))

	def _build_classifiers(self, data_dimensions, estimators):
		""" Returns an array containing all the classifiers for the ensemble """

		if self.verbose:
			helper.bprint("Initializing ensemble classifiers")

		return [
			RandomForestClassifier(n_estimators=estimators, n_jobs=-1),
			ExtraTreesClassifier(n_estimators=estimators, n_jobs=-1),
			XGBoostClassifier(n_estimators=estimators+ 500),
		], 3

	def _get_sets(self, train, test):
		"""
		Return a test training set split

		First returns the x train test, then y
		"""
		return self.x[train], self.x[test], self.y[train], self.y[test]

	def _blend(self, train, test):
		""" Blend all the classifiers together using logistic regression """

		if self.verbose:
			helper.bprint("Blending classifiers")

		log_regressor = LogisticRegression(solver='lbfgs')
		log_regressor.fit(train, self.y)
		return log_regressor.predict_proba(test)

	def _save(self, submission):
		""" Saves the final submission to csv using the helper module"""

		if self.verbose:
			helper.gprint("Saving data to csv")
		
		helper.save_submission_csv(submission, "ensemble")

	def _stretch(self, y):
		return (y - y.min()) / (y.max() - y.min())
