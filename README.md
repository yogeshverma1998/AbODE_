# AbODE: Ab initio Antibody Design using Conjoined ODEs
Evaluation and pre-trained models.

## Prerequisites

- torchdiffeq : https://github.com/rtqichen/torchdiffeq.
- pytorch >= 1.10.0
- torch-scatter 
- torch-sparse 
- torch-cluster 
- torch-spline-conv 
- torch-geometric 
- astropy
- networkx


## Datasets
Datasets are placed in "data/" folder. Note that [Structural Antibody Dataset](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab) is an evolutionaey dataset in which new structures are added each week. The model has been evaluated on a same specification as other competing methods in benchmarks.

## Evaluation:

Unconditional Antibody Sequence and structure generation:
```
python evaluation_uncond.py --cdr ${CDR-region-antibody}(1/2/3)
```

Conditional Antibody Sequence and structure generation:
```
python evaluation_cond.py --cdr ${CDR-region-antibody}(1/2/3)
```
