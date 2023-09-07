# How to produce paper results

## NN runs:

For all NN runs, the ensembling is performed externally, producing ROC curve files for an ensemble of 10 and 50 NNs. For this do: 

```
    python do_ensemble_NN.py --directory "results/NN_to_be_ensembled/"
```
## Different runs for paper:

### Baseline performances:

```
    python run_pipeline.py --mode "supervised" --classifier "NN" --directory "results/supervised_baseline_NN/" --input_set "baseline"
    python run_pipeline.py --mode "supervised" --classifier "BDT" --directory "results/supervised_baseline_BDT/" --input_set "baseline"
    python run_pipeline.py --mode "IAD" --classifier "NN" --directory "results/IAD_baseline_NN/" --input_set "baseline"
    python run_pipeline.py --mode "IAD" --classifier "BDT" --directory "results/IAD_baseline_BDT/" --input_set "baseline"
```

### Gaussian inputs for classifier type X and N gaussian inputs:

```
    python run_pipeline.py --mode "IAD" --classifier X --directory "results/IAD_X_NG/" --input_set "baseline" --gaussian_inputs N
```

### All other sets for classifier X and input_set A:

```
    python run_pipeline.py --mode "IAD" --classifier X --directory "results/IAD_X_A/" --input_set A
```

### 1D scan


### 2D scan