rule all:
    input:
        ["results/{dataset}/total/{k}/{lambda}/logpmf.txt" for k in range(2, 15)
         for lambda in [0, 0.0025, 0.05, 0.12, 0.2] for dataset in "prostate"]

rule cv:
    input:
        "{dataset}/{fold}/train_m.csv"
    output:
        "results/{dataset}/{fold}/{k}/{lambda}/e.csv",
        "results/{dataset}/{fold}/{k}/{lambda}/s.csv",
    script:
        "scripts/ten_fold.py"

rule test_train_split:
    input:
        "{dataset}/M/m.csv"
    output:
        trains = ["{dataset}/{fold}/train_m.csv" for fold in range(10)],
        tests = ["{dataset}/{fold}/test_m.csv" for fold in range(10)]
    script:
        "scripts/test_train_split.py"

rule evaluate:
    input:
        "{dataset}/{fold}/test_m.csv",
        "results/{dataset}/{fold}/{k}/{lambda}/s.csv",
    output:
        "results/{dataset}/{fold}/{k}/{lambda}/e_new.csv",
        "results/{dataset}/{fold}/{k}/{lambda}/logpmf.txt"
    script:
        "scripts/evaluate.py"


rule ten_fold:
    input:
        [f"results/{dataset}/{{fold}}/{k}/{lambda}/logpmf.txt" for fold in range(10)]
    output:
        "results/{dataset}/total/{k}/{lambda}/logpmf.txt"
    run:
        open(output[0], "w").write(
            sum(float(open(f).read()) for f in input)/10)
