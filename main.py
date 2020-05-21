import helper
import argparse
from ensemble import Ensemble

if __name__ == '__main__':
	argument_parser = argparse.ArgumentParser()
	argument_parser.add_argument("-e", "--estimators", required=True,
									help="Number of estimators to use for the classifiers")
	argument_parser.add_argument("-l", "--layers", required=True, help="Number of hidden layer")
	arguments = vars(argument_parser.parse_args())

	ensemble = Ensemble(estimators=int(arguments['estimators']), hidden_layers=int(arguments['layers']))
	ensemble.run()