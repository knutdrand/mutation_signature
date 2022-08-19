# Mutation Signature
# from scripts.split_fasta_regions import split_fasta
# from snakemake.utils import R

outDir = "data"   # symlink to /work
refDir = "ref" 	  # symlink to /work
SCRATCH  = "/work/scratch/sasha"

# SAMPLES, = glob_wildcards(outDir + "/reads/{sample}-R1_001.fastq.gz")

localrules: all
	
rule all:
	input: INPUT_ALL

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
