import numpy as np
def gauss1d(v, sigma):
    if sigma <= 0: return v.copy()
    r = int(3*sigma)
    x = np.arange(-r, r+1)
    k = np.exp(-x**2/(2*sigma*sigma)); k /= k.sum()
    pad = np.pad(v, (r, r), mode='edge')
    return np.convolve(pad, k, mode='valid')
