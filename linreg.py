#!/usr/bin/env python3

# imports
#import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import statsmodels.api as sm
import statsmodels.formula.api as smf

# migrate sklearn dict data to DF
boston = load_boston()
#diagnostics
#print(boston.data.shape)
#print(boston.keys())
bostonDF = pd.DataFrame(boston.data)
bostonDF.columns = boston.feature_names
bostonDF['MEDV'] = boston.target
# diagnostics
#print(bostonDF.head())

# run fit
results = smf.ols('MEDV ~ LSTAT', data=bostonDF).fit()
print(results.summary())

