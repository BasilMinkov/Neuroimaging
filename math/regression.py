import numpy as np
import scipy.optimize

from .curves import sine_wave

def sin_regression(x, y):
    '''
    Fit sine to the input time sequence, and return fixing parameters 
    "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"

    Parameters
    ----------
    x : list
    y : list

    Returns
    -------

    '''
    
    # make numpy array if list
    x = np.array(x)
    y = np.array(y)
    ff = np.fft.fftfreq(len(x), (x[1]-x[0])) # assume uniform spacing of argument
    Fy = abs(np.fft.fft(y))
    guess_freq = abs(ff[np.argmax(Fy[1:])+1]) # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(y) * 2.**0.5
    guess_offset = np.mean(y)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    popt, pcov = scipy.optimize.curve_fit(sine_wave, x, y, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}