# full version of ccovf
# raw function return (np.correlate(xo, yo, 'full') / d)[n - 1:]
import numpy as np


def ccovf(x, y, unbiased=True, demean=True):
    ''' crosscovariance for 1D

    Parameters
    ----------
    x, y : arrays
       time series data
    unbiased : boolean
       if True, then denominators is n-k, otherwise n

    Returns
    -------
    ccovf : array
        autocovariance function

    Notes
    -----
    This uses np.correlate which does full convolution. For very long time
    series it is recommended to use fft convolution instead.
    '''
    n = len(x)
    if demean:
        xo = x - x.mean()
        yo = y - y.mean()
    else:
        xo = x
        yo = y
    if unbiased:
        xi = np.ones(n)
        d = np.correlate(xi, xi, 'full')
    else:
        d = n
    return np.correlate(xo, yo, 'full') / d


def ccf(x, y, unbiased=True):
    '''cross-correlation function for 1d

    Parameters
    ----------
    x, y : arrays
       time series data
    unbiased : boolean
       if True, then denominators for autocovariance is n-k, otherwise n

    Returns
    -------
    ccf : array
        cross-correlation function of x and y

    Notes
    -----
    This is based np.correlate which does full convolution. For very long time
    series it is recommended to use fft convolution instead.

    If unbiased is true, the denominator for the autocovariance is adjusted
    but the autocorrelation is not an unbiased estimtor.

    '''
    cvf = ccovf(x, y, unbiased=unbiased, demean=True)
    return cvf / (np.std(x) * np.std(y))
