import helper

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

class Ensemble():
	"""Wrapper around the main algorithm"""

	def __init__(self, folds=10, verbose=True):
		self.folds = folds
		self.verbose = verbose
		self._load_data()

	def _load_data(self):
		""" Loads the data using the helper """

		if self.verbose:
			helper.bprint("Loading data from csv.")

		self.x, self.y, self.x_test = helper.load_data()

	def stretch(self, y):
		return (y - y.min()) / (y.max() - y.min())
		
	def _build_classifiers(self, data_dimensions):
		""" Returns an array containing all the classifiers for the ensemble """
		if self.verbose:
			helper.bprint("Initializing ensemble classifiers")

		return [
      RandomForestClassifier(n_estimators=100, n_jobs=-1),
      ExtraTreesClassifier(n_estimators=100, n_jobs=-1)
    ]

	def run(self):
		models = self._build_classifiers(0)
