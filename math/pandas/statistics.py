import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr, spearmanr
from statsmodels.sandbox.stats.multicomp import multipletests


def standardize(df):
    '''
    Standardization rescales data to have a mean (mu) of 0 
    and standard deviation (sigma) of 1 (unit variance).
	
	Equation
	--------
    $x_{changed} = \\frac{x - \\mu}{\\sigma}$

    Parameters
    ----------
    df : pandas.DataFrame, numpy.array
	
    '''
    return (df - df.mean()) / df.std()


def normalize(df):
    '''
	Normalization rescales the values into a range of [0,1]. 
	This might be useful in some cases where all parameters need 
	to have the same positive scale. However, the outliers from 
	the data set are lost.

	Equation
	--------
    $x_{changed} = \\frac{x - x_{min}}{x_{max} - x_{min}}$

    Parameters
    ----------
    df : pandas.DataFrame, numpy.array


    '''
    return (df - df.min()) / (df.max() - df.min())


def rmse(df_1, df_2):
	'''
	The root-mean-square error (RMSE) is a frequently used measure of 
	the differences between values (sample or population values) predicted by 
	a model or an estimator and the values observed. The RMSD represents 
	the square root of the second sample moment of the differences between 
	predicted values and observed values or the quadratic mean of these differences. 
	These deviations are called residuals when the calculations are performed over 
	the data sample that was used for estimation and are called errors (or prediction errors) 
	when computed out-of-sample. The RMSD serves to aggregate the magnitudes of the errors in 
	predictions for various times into a single measure of predictive power. 
	RMSD is a measure of accuracy, to compare forecasting errors of different models 
	for a particular dataset and not between datasets, as it is scale-dependent.

	Equation
	________
	$RMSE = $

	Reference
	_________
	1.	Hyndman, Rob J.; Koehler, Anne B. (2006). "Another look at measures of forecast accuracy". 
		International Journal of Forecasting. 22 (4): 679–688. CiteSeerX 10.1.1.154.9771. 
		doi:10.1016/j.ijforecast.2006.03.001.

    Parameters
    ----------
    df1 : pandas.DataFrame, numpy.array
    df2 : pandas.DataFrame, numpy.array

	'''

	return ((df_1 - df_2)**2).mean()**0.5

def moving_correlation(time_array, x, y, gap, correlation="positive"):

    # set size of the rolling window 
    p = np.zeros(x.shape[0]-gap) # p-value vector
    c = np.zeros(x.shape[0]-gap) # statistic vector

    # calculate the rolling corelation coefficient
    for i in np.arange(x.shape[0]-gap):
        ans = spearmanr(x[i:i+gap], y[i:i+gap])
        c[i] = ans[0] 
        p[i] = ans[1]

    # adjust p-values with bonferroni correction
    p_adjusted = multipletests(p, method='bonferroni')[1]

    # preper arrays to store info about significance (1 - significant, 0 - non-significant)
    c1 = np.zeros(x.shape[0]-gap)
    c2 = np.zeros(x.shape[0]-gap)
    c3 = np.zeros(x.shape[0]-gap)
    
    # 
    c_green = np.zeros(x.shape[0]-gap)
    c_yellow = np.zeros(x.shape[0]-gap)
    c_red = np.zeros(x.shape[0]-gap)

    # choose correlation peaking type
    if correlation == "absolute":
    	corr = lambda x : np.abs(x)
    elif correlation == "positive":
    	corr = lambda x: x
    elif correlation == "negative":
    	corr = lambda x: -x

    # write info about significance
    p_threshold = 5e-07 
    c1[(p_adjusted < p_threshold) & (corr(c) >= 0.8)] = 1
    c2[(p_adjusted < p_threshold) & (corr(c) >= 0.6) & (corr(c) < 0.8)] = 1
    c3[(p_adjusted < p_threshold) & (corr(c) >= 0.4) & (corr(c) < 0.6)] = 1
    c1[c1 < 1] = 0
    c2[c2 < 1] = 0
    c3[c3 < 1] = 0
    
    num = [c1, c2, c3]
    col = [c_green, c_yellow, c_red]

    z = list(zip(num, col))

    counter = 0

    for unit in z:
        for i in range(len(unit[0])):
            if unit[0][i] == 0:
                if counter == 0:
                    unit[1][i] = np.nan
                else:
                    unit[1][i] = time_array[i]
                    counter -= 1
            else:
                unit[1][i] = time_array[i]
                counter = gap
    
    z[2][1][z[0][1] == z[2][1]] = np.nan
    z[2][1][z[1][1] == z[2][1]] = np.nan
    z[1][1][z[0][1] == z[1][1]] = np.nan

    return z

