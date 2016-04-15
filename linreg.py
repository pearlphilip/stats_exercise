#!/usr/bin/env python3

# imports
#import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from sklearn.datasets import load_boston
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import abline_plot

PDF='linreg.pdf'

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

# plot
#plt.plot()
with PdfPages(PDF) as pdf:
    fig = abline_plot(model_results=results)
    ax = fig.axes[0]
    ax.scatter(bostonDF['LSTAT'], bostonDF['MEDV'])
    ax.margins(.1)
    pdf.savefig()
    plt.close()
