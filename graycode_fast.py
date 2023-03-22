import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multinomial
import random
from tqdm import tqdm
import itertools
data = pd.read_csv("law.csv")

from math import factorial

def gray_code_composition(n: int, save:bool=False) -> np.ndarray:
    length = int(factorial(2*n-1)/(factorial(n-1)*factorial(2*n-1-(n-1))))
    compositions = np.zeros((length, n))

    # first combination
    curr = np.zeros(n)
    curr[0] = n
    compositions[0, :] = curr
    p = 0
    pos=1

    while (curr[n-1] != n): # we are done once n is in last
        if p == 0:
            if np.count_nonzero(curr) > 1:
                b = np.flatnonzero(curr)[1]
            else:
                b=0
            if b == 1:
                if curr[0] == 1:
                    p = 1
            elif (n - curr[0]) % 2 == 0:
                d, i, p = 0, 1, 1
            elif curr[b] % 2 == 1:
                d, i, p = 0, b, b
            else:
                i, d = 0, b
        else:
            if (n - curr[p]) % 2 == 1:
                d, i = p, p-1
                if curr[p] % 2 == 0:
                    i = 0
                p = i
            elif curr[p+1] % 2 == 0:
                i, d = p+1, p
                if curr[p] == 1:
                    p= p+1
            else:
                i, d = p, p+1
        curr[i] +=1
        curr[d] -= 1
        if curr[0] > 0:
            p=0
        compositions[pos] = curr
        pos += 1
    if save:
        np.save("gray_codes.npy", compositions)
    return compositions


n = data.shape[0]
# To run and save graycodes on first execution:
# graycodes = gray_code_composition(n, True)
graycodes = np.load("gray_codes.npy")

correlations = data['LSAT'] * data['GPA']
lsat_square = data['LSAT']**2
gpa_square = data['GPA']**2

corrs = []

for graycode in tqdm(graycodes, total=graycodes.shape[0]):
    # formula 
    # numerator n* sum (xi*yi) - sum(x)*sum(y)
    curr_x = np.sum(graycode*data['LSAT'])
    curr_y = np.sum(graycode*data['GPA'])
    corr = n*np.sum(np.dot(graycode, correlations)) - curr_x*curr_y
    # denominator
    # sqrt(n* sum (x_square)-sum(x)**2) * sqrt(n*sum(y_square)-sum(y)**2)
    corr = corr/(np.sqrt(n* np.sum(np.dot(graycode, lsat_square))-curr_x**2)*np.sqrt(n * np.sum(np.dot(graycode, gpa_square))-curr_y**2))
    corrs.append(corr)


# save 
np.save("graycode_enumeration_corrcoefs.npy", np.array(corrs))
