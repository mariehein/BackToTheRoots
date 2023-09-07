# How to produce paper results

## NN runs:

For all NN runs, the ensembling is performed externally, producing ROC curve files for an ensemble of 10 and 50 NNs. For this do: 

```
    python do_ensemble_NN.py --directory "results/NN_to_be_ensembled/"
```
## Different runs for paper:

### Baseline performances:

```
    python run_pipeline.py --mode "supervised" --classifier "NN" --input_set "baseline" --directory "results/supervised_baseline_NN/"
    python run_pipeline.py --mode "supervised" --classifier "BDT" --input_set "baseline" --directory "results/supervised_baseline_BDT/"
    python run_pipeline.py --mode "IAD" --classifier "NN" --input_set "baseline" --directory "results/IAD_baseline_NN/"
    python run_pipeline.py --mode "IAD" --classifier "BDT" --input_set "baseline" --directory "results/IAD_baseline_BDT/"
```

### Gaussian inputs for classifier type X and N gaussian inputs:

Performed for:
- X in ["NN", "BDT"]
- N in [1, 2, 5, 10, 30, 50]

```
    python run_pipeline.py --mode "IAD" --classifier X --input_set "baseline" --gaussian_inputs N --directory "results/IAD_X_NG/" 
```

### All other sets for classifier X and input_set A:

Performed for:
- X in ["BDT", "NN"]
- A in ["extended1", "extended2", "extended3"] (note "baseline" already performed for baseline performance comparisons)

```
    python run_pipeline.py --mode "IAD" --classifier X --input_set A --directory "results/IAD_X_A/" 
```

### 1D scan for classifier X and input_set A:

Performed for:
- X="BDT" and A in ["baseline", "extended3", "kitchensink"]
- X="NN" and A="baseline"

```
    sig = (0 100 200 300 400 500 750 1000 1200 1500 2000)
    for s in ${sig}; do
        python run_pipeline.py --mode "IAD" --classifier X  --input_set A  --randomize_seed --signal_number ${s} --directory "results/1D_Nsig_X_A/Nsig_${s}/"
    done
```

### 2D scan

Performed for:
- X="BDT" and A in ["baseline", "kitchensink"]

```
    bkg=(25000 50000 75000 100000 125000 150000 175000 200000)
    sig=(0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5)

    for b in ${bkg}; do
        for s in ${sig} do
            python run_pipeline_2D.py --mode "IAD" --classifier X --input_set A --randomize_seed --signal_significance ${s} --N_CR ${b} --N_bkg ${b} --directory "results/2D_X_A/${b}_${b}_${s}/"
            sbatch 2D_scan_IAD.slurm ${bkg[i]} ${bkg[i]} ${sig[j]} "${bkg[i]}_${bkg[i]}_${sig[j]}/"
        done
    done
```