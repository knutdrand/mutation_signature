# import snakemake
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
M = pd.read_csv(snakemake.input[0], header=None, delimiter="\t")
for i in range(10):
    print(i)
    train, test = train_test_split(M, test_size=0.3)
    # print(train.to_numpy())
    print(snakemake.output)
    train.to_csv(snakemake.output.trains[i], sep="\t", header=False)
    test.to_csv(snakemake.output.tests[i], sep="\t", header=False)
