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
# our test to see how linked NOX and INDUS were
# this test showed some increase in INDUS relevance
# when NOX was dropped
#bostonDF.drop('NOX', axis=1, inplace=True)
# test to see how modifying the scale of an input
# impacts the final coeffecient
#print(bostonDF['CRIM'])
#bostonDF['CRIM'] /= 100000
cols = bostonDF.columns[:-1]
formula = 'MEDV ~ ' + cols[0]
for col in cols[1:]:
    formula = formula + ' + ' + col
print(formula)
results = smf.ols(formula, data=bostonDF).fit()
print(results.summary())

'''
# plot
#plt.plot()
with PdfPages(PDF) as pdf:
    fig = abline_plot(intercept=results.params['Intercept'],
                        slope=results.params['LSTAT'])
    ax = fig.axes[0]
    ax.scatter(bostonDF['LSTAT'], bostonDF['MEDV'])
    #plt.title('MEDV vs. LSTAT + AGE (LSTAT only)')
    ax.margins(.1)
    pdf.savefig()
    plt.close()

    fig = abline_plot(intercept=results.params['Intercept'],
                        slope=results.params['AGE'])
    ax = fig.axes[0]
    ax.scatter(bostonDF['AGE'], bostonDF['MEDV'])
    ax.margins(.1)
    #plt.title('MEDV vs LSTAT + AGE (AGE only)')
    pdf.savefig()
    plt.close()
'''
