import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import helper
import time

def param_tuning(params):
  tic = time.perf_counter()

  # Load data using helper
  X, y, submission_data = helper.load_data(False)

  # Create model and set estimator grid
  # Use GPU for faster results
  model = xgb.XGBRegressor(tree_method="gpu_hist", gpu_id=0,objective='binary:logistic',booster='gbtree',eval_metric='mlogloss', n_estimators=1000)

  # Execute the grid search
  helper.bprint('Starting grid search')
  kfold = StratifiedKFold(n_splits=10, shuffle=True)
  grid_search = GridSearchCV(model, params, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold)
  grid_result = grid_search.fit(X, y)

  # Print best result
  helper.gprint("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

  toc = time.perf_counter()
  helper.bprint(f"Hyper parameter tuned in {toc - tic:0.4f} seconds")

# Tune max_depth and min_child_weight
#param_tuning({
# 'max_depth':range(3,10,1),
# 'min_child_weight':range(1,6,1)
#})

# Tune gamma
# using optimal parameters found before
#param_tuning({
#'max_depth':[6],
#'min_child_weight':[4],
# 'gamma':[i/10.0 for i in range(0,5)]
#})

# Tune subsample and colsample_bytree
# using optimal parameters found before
#param_tuning({
#'max_depth':[6],
#'min_child_weight':[4],
#'gamma': [0.3],
#'subsample':np.arange(0.8, 1.0, 0.01),
#'colsample_bytree':np.arange(0.5, 0.6, 0.01)
#})

# Tune regularization
# using optimal parameters found before
param_tuning({
'max_depth':[6],
'min_child_weight':[4],
'gamma': [0.3],
'subsample':[0.98],
'colsample_bytree':[0.52],
'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
})