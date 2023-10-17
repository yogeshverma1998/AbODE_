# AbODE: Ab initio Antibody Design using Conjoined ODEs
Evaluation and pre-trained models.

## Prerequisites

- torchdiffeq : https://github.com/rtqichen/torchdiffeq.
- pytorch >= 1.10.0
- torch-scatter 
- torch-sparse 
- torch-cluster 
- torch-spline-conv 
- torch-geometric == 2.0.4
- astropy
- networkx
- tqdm
- Biopython

Make sure to have the correct version of each library as some errors might arise due to version-mismatch. The libraries-version of the local conda env is in `env_list.txt` 
## Datasets
Datasets are placed in the "data/" folder. Note that [Structural Antibody Dataset](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab) is an evolutionary dataset in which new structures are added each week. The model has been evaluated on the same specification as other competing methods in benchmarks.

## Evaluation:

Unconditional Antibody Sequence and structure generation:
```
python evaluation_uncond.py --cdr ${CDR-region-antibody}(1/2/3)
```

Conditional Antibody Sequence and structure generation:
```
python evaluation_cond.py --cdr ${CDR-region-antibody}(1/2/3)
```
