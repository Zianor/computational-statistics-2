import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multinomial
import random
from tqdm import tqdm
import itertools
data = pd.read_csv("law.csv")[:5]

from math import factorial

def gray_code_composition(n: int, save:bool=False) -> np.ndarray:
    """N-W algorithm to run through the compositions
    """
    length = int(factorial(2*n-1)/(factorial(n-1)*factorial(2*n-1-(n-1))))
    gray_codes = np.zeros((length, n))
    pbar = tqdm(total=length)

    # first combination
    curr = np.zeros(n)
    curr[0] = n
    gray_codes[0, :] = curr

    value_first_nonzero = n
    i = 1
    while (curr[n-1] != n): # we are done once n is in last
        pbar.update(1)
        if value_first_nonzero != 1:
            first_nonzero = 0
        else:
            first_nonzero += 1
        value_first_nonzero = curr[first_nonzero]
        curr[first_nonzero] = 0
        curr[0] = value_first_nonzero - 1
        curr[first_nonzero+1] = curr[first_nonzero+1] +1
        gray_codes[i, :] = curr
        i += 1
    if save:
        np.save("gray_codes.npy", gray_codes)
    return gray_codes


n = data.shape[0]
# To run and save graycodes on first execution:
# graycodes = gray_code_composition(n, True)
graycodes = np.load("gray_codes.npy")

lsat_mean = np.mean(data['LSAT'])
gpa_mean = np.mean(data['GPA'])

correlation = data['LSAT'] * data['GPA']
lsat_square = (data['LSAT'] - lsat_mean)**2
gpa_square = (data['GPA'] - gpa_mean)**2

calc_cor = np.zeros((graycodes.shape[0], 2))


corrs = []
mltn = multinomial(n, [1/n]*n) 
for i, graycode in tqdm(enumerate(graycodes), total=graycodes.shape[0]):
    calc_cor[i, 0] = (np.dot(graycode, correlation) - 
                      (np.sum(graycode*data['LSAT'])*np.sum(graycode*data['GPA']))/n)
    calc_cor[i, 0] = calc_cor[i, 0]/((np.dot(graycode, lsat_square))*(np.dot(graycode, gpa_square)))

    # calculate probability of graycode for later weighting
    calc_cor[i, 1] = mltn.pmf(graycode.astype(np.int32))


# save 
np.save("graycode_enumeration_corrcoefs.npy", calc_cor)
