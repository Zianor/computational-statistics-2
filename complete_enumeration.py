from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
df = pd.read_csv("law.csv")
n = len(df)
enumeration = list(combinations_with_replacement(list(range(n)), n))

data = df.values
corrcoefs = []
for e in tqdm(enumeration):
    c = np.corrcoef(data[list(e)].T)
    corrcoefs.append(c[0][1])

pd.DataFrame(corrcoefs).to_csv("complete_enumeration_corrcoefs.csv", index=False)
