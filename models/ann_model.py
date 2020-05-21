import keras
import numpy as np
import keras.utils.np_utils as util
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input

class ANNClassifier():
	""" Wrapper around Keras artificial neural network, adhering to partial scikit interface """

	def __init__(self, input_size, classes=2, neurons=888, hidden_layers=3):
		self.input_size = input_size
		self.classes = classes
		self.neurons = neurons
		self.hidden_layers = hidden_layers

		self._build()

	def __str__(self):
		return "ANNClassifier"

	def _build(self):
		""" Constructs the sequential model """

		self.model = Sequential()
		
		# Add first layer
		self.model.add(Dense(self.neurons, input_dim=self.input_size, activation='relu'))

		# Add hidden layers
		for i in range(1, self.hidden_layers + 1):
			neurons = self.neurons // (i * 2)
			self.model.add(Dense(neurons, activation='relu'))
			self.model.add(Dropout(0.2))

		# Add final activation layer and compile model
		self.model.add(Dense(self.classes,activation='sigmoid'))
		self.model.compile(loss='categorical_crossentropy', optimizer='adam')

	def fit(self, x, y):
		y = util.to_categorical(y, 2)
		self.model.fit(x, y)

	def predict_proba(self, x):
		return self.model.predict_proba(x)