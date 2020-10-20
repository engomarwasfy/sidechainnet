_!! Note: At this moment, SidechainNet's API is being finalized and documentation may be incorrect. 10/16/2020 !!_

SidechainNet
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/jonathanking/sidechainnet.svg?branch=master)](https://travis-ci.com/jonathanking/sidechainnet)

SidechainNet is a protein structure prediction dataset that directly extends [ProteinNet](https://github.com/aqlaboratory/proteinnet)<sup>1</sup> by Mohammed AlQuraishi.

Specifically, SidechainNet adds measurements for protein angles and coordinates that describe the complete, all-atom protein structure (backbone *and* sidechain, excluding hydrogens) instead of the protein [backbone](https://foldit.fandom.com/wiki/Protein_backbone) alone.

**This repository provides the following:**
1. SidechainNet datasets stored as pickled Python dictionaries (accessible via `sidechainnet.load`).
2. Methods for loading and batching SidechainNet data efficiently in PyTorch. 
3. Methods for generating protein structure visualizations (`.pdb`, [`3Dmol`](http://3dmol.csb.pitt.edu), `.gltf`) from model predictions.
 
 **Summary of SidechainNet data**
 
| Entry | Dimensionality* | Label in SidechainNet data | ProteinNet | SidechainNet | 
| :---: | :---: |  :---: | :---: | :---: | 
| Primary sequence | *L x 1* | `seq` | X | X | 
| Secondary Structure^ | *L x 8* |  `sec` | X | X | 
| [PSSM](https://en.wikipedia.org/wiki/Position_weight_matrix) + Information content | *L x 21* |  `evo` | X | X | 
| Missing residue mask | *L x 1* |  `msk` | X | X | 
| Backbone coordinates | *L x 4<sup>⸸</sup> x 3* |  `crd`, subset `[0:4]` | X | X | 
| Backbone torsion angles | *L x 3* |  `ang`, subset `[0:3]` | . | X | 
| Backbone bond angles | *L x 3* |  `ang`, subset `[3:6]` | . | X | 
| Sidechain torsion angles | *L x 6* |   `ang`, subset `[6:12]` | . | X | 
| Sidechain coordinates | *L x 10 x 3* |  `crd`, subset `[4:14]` | . | X | 

**L* reperesents the length of any given protein in the dataset.

^[Currently unsupported](https://github.com/aqlaboratory/proteinnet/issues/5) in ProteinNet and, therefore, unsupported in SidechainNet.

<sup>⸸</sup>SidechainNet explicitly includes oxygen atoms as part of the backbone coordinate data in contrast to ProteinNet, which only includes the primary `N, C_alpha, C` atoms.

## Installation
To run this code, it's recommended to first perform a developmental install of the package with pip in your current environment with `pip install -e .`. This will install the `sidechainnet` package in your environment.

## Downloading Data
For every existing ProteinNet dataset, a corresponding SidechainNet dataset has been created and can be accessed via `sidechainnet.load`. 

There are separate datasets for each available CASP competition (CASP 7-12) as well as each "thinning" of the data as described by ProteinNet (`30, 50, 70, 90, 95, 100%`). A thinning represents the same dataset but has been clustered and downsampled to reduce its size. Thinnings marked as `100` contain the complete dataset.

Files are named using the following convention: `casp{CASP_NUMBER}_{THINNING}.pkl`.

## Usage Examples

### Loading SidechainNet as a Python dictionary

```python
import pickle
with open("casp12_100.pkl", "rb") as f:
    data = pickle.load(f)
```
In its most basic form, SidechainNet is stored as a Python dictionary organized by train, validation, and test splits. These splits are identical to those described in ProteinNet (note the existence of multiple validation sets).
 
 Within each train/validation/test split in SidechainNet is another dictionary mapping data entry types (`seq`, `ang`, etc.) to a list containing this data type for every protein. In the example below, `seq{i}`, `ang{i}`, ... all refer to the `i`<sup>th</sup> protein in the dataset.
```python
data = {"train": {"seq": [seq1, seq2, ...],
                  "ang": [ang1, ang2, ...],
                  "crd": [crd1, crd2, ...],
                  "evo": [evo1, evo2, ...],
                  "sec": [sec1, sec2, ...],
                  "ids": [id1, id2, ...],          # Corresponding ProteinNet IDs
                  },
        "valid-30": {...},
            ...
        "valid-90": {...},
        "test": {...},
        "settings": {...}
        }
```

### Using SidechainNet to train an all-atom protein structure prediction model 

For a complete example of model training, please see [sidechainnet/examples/train.py](./sidechainnet/examples/train.py). Below is an outline of how to use this repository to load SidechainNet data with our custom DataLoaders and train a model.

```python
import sidechainnet
from sidechainnet.examples.rgn import RGN
from sidechainnet.examples.losses import drmsd


train, train_eval, validations, test = sidechainnet.get_dataloaders("casp12_100.pkl")
model = RGN()

for epoch in range(10):

    # Training epoch
    for seq, tgt_angles, tgt_coords in train:
        pred_coords = model(seq)
        loss = drmsd(pred_coords, tgt_coords)
        loss.backwards()
        ...
    
    # Evaluate performance on downsampled training set
    for seq, tgt_angles, tgt_coords in train_eval:
        pred_coords = model(seq)
        train_loss = drmsd(pred_coords, tgt_coords)
        ...

    # Evaluate performance on each of the 7 validation sets
    for validation_set in validations:
        for seq, tgt_angles, tgt_coords in validations:
            pred_coords = model(seq)
            val_loss = drmsd(pred_coords, tgt_coords)
            ...

# Evaluate performance on test set
for seq, tgt_angles, tgt_coords in test:
        pred_coords = model(seq)
        test_loss = drmsd(pred_coords, tgt_coords)
    ...
```

### Other included utilities
In addition to the data itself, this repository also provides several utilities:
- `sidechainnet.get_dataloaders` (uses `sidechainnet.utils.dataset.BinnedProteinDataset` and `sidechainnet.utils.dataset.SimilarLengthBatchSampler`)
    - By using these together to create a PyTorch Dataloader, we can handle the batching of SidechainNet data intelligently and increase training speed.
     - Each batch contains proteins of similar lengths but the average length for a batch is chosen at random from a bin. Using batches of similar lengths allows the computation of DRMSD to be parallelized effectively and improves performance.
- `PdbBuilder`
    - Generates structure files (`.pdb`) from model predictions.
    - Also enables the creation of 3D-object files (`.gltf`) in order to log this data using Weights and Biases ([example](https://app.wandb.ai/koes-group/protein-transformer/reports/Evaluating-the-Impact-of-Sequence-Convolutions-and-Embeddings-on-Protein-Structure-Prediction--Vmlldzo2OTg4Nw)).
- Miscelaneous utilities for training protein structure prediction models
    - Batch parallelized DRMSD computation.

## Directions to Reproduce SidechainNet

If you are only interested in using and interacting with SidechainNet data, please see the above examples. However, if you would like to reproduce our work or make modifications to the dataset, you may follow the directions below to generate SidechainNet from scratch.

[How to reproduce and generate SidechainNet](how_to_reproduce.md)

## Package Requirements

- Python 3
- ProDy (`pip install ProDy`)
- Biopython
- numpy
- scipy
- [PyTorch](https://pytorch.org/get-started/locally/)
- tqdm


## Acknowledgements

Many thanks to Mohammed AlQuraishi for his inspiring work on protein structure prediction. Thanks, also, to [Jeppe Hallgren](https://github.com/JeppeHallgren) for his development of a ProteinNet text record [parser](https://github.com/biolib/openprotein/blob/master/preprocessing.py), which I have used in part here.

This work is supported by R01GM108340 from the National Institute of General Medical Sciences, is supported in part by the University of Pittsburgh Center for Research Computing through the resources provided, and by NIH T32 training grant T32 EB009403 as part of the HHMI-NIBIB Interfaces Initiative.

Project structure (continuous integration, docs, testing) based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.

## References
1. [End-to-End Differentiable Learning of Protein Structure](https://doi.org/10.1016/j.cels.2019.03.006). AlQuraishi, Mohammed. Cell Systems, Volume 8, Issue 4, 292 - 301. (2019).
2. [ProteinNet: a standardized data set for machine learning of protein structure.](https://doi.org/10.1186/s12859-019-2932-0). AlQuraishi, Mohammed. BMC Bioinformatics 20, 311 (2019).
 

## Copyright

Copyright (c) 2020, Jonathan King
