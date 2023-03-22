import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

df = pd.read_csv("law.csv")

from itertools import combinations_with_replacement
from scipy.stats import multinomial
from joblib import Parallel, delayed

def complete_enumeration_bootstrap(f, data):
    n = len(data)
    # get all possible combinations of the data
    combinations = list(combinations_with_replacement(range(n), n))
    # calculate the statistic for each combination
    distribution = np.array([f(data[list(c)]) for c in tqdm(combinations)])
    # calculate the probabilities of each combination
    probabilities = [multinomial.pmf(np.bincount(c, minlength=n), n, [1/n]*n) for c in tqdm(combinations)]
    return distribution, probabilities

def complete_enumeration_bootstrap_parallelized(f, data, n_jobs=4):
    n = len(data)
    # get all possible combinations of the data
    combinations = list(combinations_with_replacement(range(n), n))
    # calculate the statistic for each combination
    # use batches of 1_000_000 combinations to avoid memory issues
    distribution = np.concatenate(Parallel(n_jobs=n_jobs)(delayed(lambda x: np.array([f(data[list(c)]) for c in x]))(combinations[i:i+1000_000]) for i in tqdm(range(0, len(combinations), 1000_000))))
    # calculate the probabilities of each combination
    probabilities = np.concatenate(Parallel(n_jobs=n_jobs)(delayed(lambda x: np.array([multinomial.pmf(np.bincount(c, minlength=n), n, [1/n]*n) for c in x]))(combinations[i:i+1000_000]) for i in tqdm(range(0, len(combinations), 1000_000))))
    return distribution, probabilities

law_complete_enumeration_hist = complete_enumeration_bootstrap_parallelized(lambda x: np.corrcoef(x.T)[0][1], df.values)
np.save("law_complete_enumeration_hist.npy", law_complete_enumeration_hist)
