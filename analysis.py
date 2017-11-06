from __future__ import print_function

import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

from scipy.optimize import curve_fit

# To get the equation to transform channel to time, run the calibrate function.
# It will return the tuple of tuples ((y, intercept), (y error, intercept error))

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def fit_gaussian(data):
    x = data['Channel'].tolist()
    y = data['Counts'].tolist()
    mean = np.mean(x)
    std = 1

    popt, pcov = curve_fit(gauss_function, x, y, p0 = [1, mean, std])

    #plt.clf()
    #plt.scatter(x,y)

    x_gauss = np.linspace(x[0], x[-1], num=1000)
    gauss = map(lambda x: gauss_function(x, popt[0], popt[1], popt[2]), x_gauss)
    #plt.plot(x_gauss, gauss)
    #plt.show()

    return(popt)

def find_gaussian(df, rng):
    stats = fit_gaussian(df[(rng[0] <= df['Channel']) & (df['Channel'] <= rng[1])])
    return stats

def find_regression(lengths, gaussian_means, gaussian_stds):
    print(lengths, gaussian_means, gaussian_stds)

    weights = 1/np.power(gaussian_stds, 2)

    # put x and y into a pandas DataFrame, and the weights into a Series
    ws = pd.DataFrame({
        'x': lengths,
        'y': gaussian_means,
        'yerr': map(lambda x: x*1, gaussian_stds)
    })

    wls_fit = sm.wls('x ~ y', data=ws, weights=1 / weights).fit()

    return((wls_fit.params['y'], wls_fit.params['Intercept']),(wls_fit.bse['y'], wls_fit.bse['Intercept']), wls_fit, ws)

def m_S(lengths, gaussian_means, gaussian_stds):
    params = find_regression(lengths, gaussian_means, gaussian_stds)
    percent_error = params[1][0] / params[0][0]
    inverse = 1/params[0][0]
    return (inverse, inverse*percent_error)
