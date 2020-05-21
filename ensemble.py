import helper


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
		
		def _build_models():
			pass

		def run():
			pass
