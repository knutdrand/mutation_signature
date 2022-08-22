import snakemake
import pandas as pd
from .mutation_sim_test import main
import numpy as np

test_m = pd.read_csv(snakemake.input[0], header=None, delimiter="\t")
S = pd.read_csv(snakemake.input[1], header=None, delimiter="\t")
E, loss = main(test_m, S)

np.savetxt(snakemake.output[0], E)
open(snakemake.output[1], "w").write(loss)
