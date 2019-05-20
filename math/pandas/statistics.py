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
    df : pandas Data Frame
	
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
    df : pandas Data Frame


    '''
    return (df - df.mean()) / df.std()


