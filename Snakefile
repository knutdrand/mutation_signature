# Mutation Signature
# from scripts.split_fasta_regions import split_fasta
# from snakemake.utils import R
import pandas as pd
import numpy as np


wildcard_constraints:
    dataset="prostate|breast_cancer",
    fold="\d+"

outDir = "data"   # symlink to /work
refDir = "ref" 	  # symlink to /work
SCRATCH  = "/work/scratch/sasha"

# SAMPLES, = glob_wildcards(outDir + "/reads/{sample}-R1_001.fastq.gz")

localrules: all
	
INPUT_ALL = [f"results/{dataset}/total/{k}/{Lambda}/logpmf.txt" for k in range(2, 15)
             for Lambda in [0, 0.0025, 0.05, 0.12, 0.2] for dataset in ["prostate", "breast_cancer"]]


rule all:
    input:
        INPUT_ALL

rule simulate:
    output:
        "results/{dataset}/m.csv"
    run:
        np.savetxt(output[0], np.random.poisson(2, 30*96).reshape(30, 96), delimiter="\t")

rule cv:
    input:
        "results/{dataset}/{fold}/train_m.csv"
    output:
        "results/{dataset}/{fold}/{k}/{Lambda}/e.csv",
        "results/{dataset}/{fold}/{k}/{Lambda}/s.csv",
    script:
        "scripts/ten_fold.py"

rule test_train_split:
    input:
        "results/{dataset}/m.csv"
    output:
        trains = ["results/{dataset}/{fold}/train_m.csv" for fold in range(10)],
        tests = ["results/{dataset}/{fold}/test_m.csv" for fold in range(10)]
    script:
        "scripts/train_test_split.py"

rule evaluate:
    input:
        "results/{dataset}/{fold}/test_m.csv",
        "results/{dataset}/{fold}/{k}/{Lambda}/s.csv",
    output:
        "results/{dataset}/{fold}/{k}/{Lambda}/e_new.csv",
        "results/{dataset}/{fold}/{k}/{Lambda}/logpmf.txt"
    script:
        "scripts/evaluate.py"


rule ten_fold:
    input:
        [f"results/{{dataset}}/{fold}/{{k}}/{{Lambda}}/logpmf.txt" for fold in range(10)]
    output:
        "results/{dataset}/total/{k}/{Lambda}/logpmf.txt"
    run:
        with open(output[0], "w") as f:
            f.write(
                sum(float(open(f).read()) for f in input)/10)


# rule A:
# 	input:
# 		read1 = outDir + "/reads/{sample}-R1_001.fastq.gz",
# 		read2 = outDir + "/reads/{sample}-R2_001.fastq.gz",
# 	output:
# 		temp(outDir + "/meta/hosts/{sample}-{q}.txt")
#	resources: mem=10, time=60*24
#	threads: 1
# 	shell:
# 		"""
# 		"""
