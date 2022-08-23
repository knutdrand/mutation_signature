import pandas as pd
import numpy as np
from mutation_sim_train import main

k = int(snakemake.wildcards.k)
Lambda = float(snakemake.wildcards.Lambda)

train_m = pd.read_csv(snakemake.input[0], header=None, delimiter="\t")
E, S = main(train_m, k, Lambda)

np.savetxt(snakemake.output[0], E)
np.savetxt(snakemake.output[1], S)
