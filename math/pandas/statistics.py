import pandas as pd

def standardize(df):
    '''
    Standardization rescales data to have a mean (mu) of 0 
    and standard deviation (sigma) of 1 (unit variance).
	
	Equation
	--------
    $x_{changed} = \\frac{x - x_{min}}{x_{max} - x_{min}}$

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
	$x_{changed} = \\frac{x - \\mu}{\\sigma}$

    Parameters
    ----------
    df : pandas.DataFrame, numpy.array


    '''
    return (df - df.mean()) / df.std()

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

