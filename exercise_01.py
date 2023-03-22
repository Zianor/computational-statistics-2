import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from itertools import combinations_with_replacement
from scipy.stats import multinomial
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d

# complete enumeration functions:
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


# monte carlo functions:
def sample(data):
    return data[random.choices(range(len(data)), k=len(data))]

def monte_carlo_bootstrap(f, data, n=40_000):
    return [f(sample(data)) for _ in tqdm(range(n))]

def monte_carlo_bootstrap_histogram(f, data, n=40_000, bins=1000):
    bins_ = np.linspace(-1, 1, bins)
    distribution = monte_carlo_bootstrap(f, data, n)
    hist, _ = np.histogram(distribution, bins=bins_)
    return hist, bins_


def plot_histogram(hist, bins, smoothing=None, ax=None, alpha=1.0, label="Distribution"):
    hist = hist / hist.sum()
    if ax is None:
        ax = plt.subplot()
    if smoothing is None:
        ax.plot(bins[:-1], hist, label=label, alpha=alpha)
    else:
        ax.plot(bins[:-1], gaussian_filter1d(hist, sigma=smoothing), label=label, alpha=alpha)


if __name__ == "__main__":
    df = pd.read_csv("law.csv")
    law_complete_enumeration = complete_enumeration_bootstrap_parallelized(lambda x: np.corrcoef(x.T)[0][1], df.values)
    np.save("law_complete_enumeration.npy", law_complete_enumeration)

    # or e.g.: 
    # distribution = monte_carlo_bootstrap(lambda x: np.corrcoef(x.T)[0][1], df.values)
