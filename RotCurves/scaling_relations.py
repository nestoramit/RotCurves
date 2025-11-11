import numpy as np


def log_Mvir_Moster2018(z, log_mstar):
    z1 = 1 + z
    n = np.power(10, 1.507 - 0.124 * z / z1)
    b = -0.621 - 0.059 * z / z1
    g = 1.055 + 0.838 * z / z1 - 3.083 * (z / z1)**2

    log_mvir = log_mstar + np.log10(0.5) + np.log10(n) + np.log10((10**(log_mstar - 10.6))**b + (10**(log_mstar - 10.6))**g)

    return log_mvir