import xgboost as xgb

class XGBoostClassifier():
	""" Wrapper around adhering to a subset of the scikit classifier interface"""

	def __init__(self, n_estimators=1500):
		self.reg = xgb.XGBRegressor(objective='multi:softprob',num_class=2, booster='gbtree',
		eval_metric='mlogloss', colsample_bytree=0.52, 
    max_depth=6, min_child_weight=4, reg_lambda=0, 
		eta=0.01, subsample=0.98, n_estimators=n_estimators, gamma=0.3, reg_alpha=1)
	
	
	def fit(self, x, y):
		self.reg.fit(x, y)


	def predict_proba(self, x):
		return self.reg.predict(x)