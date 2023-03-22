import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import itertools
data = pd.read_csv("law.csv")

from math import factorial

def gray_code_composition(n):
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
    return gray_codes


n = data.shape[0]
graycodes = gray_code_composition(n)

corrs = []
for graycode in tqdm(graycodes):
    lsat_sample = list(itertools.chain(*[[data.iloc[i]['LSAT']]*int(g) for i, g in enumerate(graycode)]))
    gpa_sample = list(itertools.chain(*[[data.iloc[i]['GPA']]*int(g) for i, g in enumerate(graycode)]))
    corr = np.corrcoef(lsat_sample, gpa_sample)[0][1]
    corrs.append(corr)


pd.DataFrame(corrs).to_csv("graycode_enumeration_corrcoefs.csv", index=False)
