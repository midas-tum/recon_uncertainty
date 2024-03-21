# Predictive uncertainty in deep learning-based MR image reconstruction

This repository contains the code for the manuscript ["Predictive uncertainty in deep learning-based MR image reconstruction using deep ensembles: Evaluation on the fastMRI data set" by Thomas Küstner, Kerstin Hammernik, Daniel Rueckert, Tobias Hepp, and Sergios Gatidis.](https://doi.org/10.1002/mrm.30030).

## Abstract

### Purpose
To estimate pixel-wise predictive uncertainty for deep learning–based MR image reconstruction and to examine the impact of domain shifts and architecture robustness.

### Methods
Uncertainty prediction could provide a measure for robustness of deep learning (DL)–based MR image reconstruction from undersampled data. DL methods bear the risk of inducing reconstruction errors like in-painting of unrealistic structures or missing pathologies. These errors may be obscured by visual realism of DL reconstruction and thus remain undiscovered. Furthermore, most methods are task-agnostic and not well calibrated to domain shifts. We propose a strategy that estimates aleatoric (data) and epistemic (model) uncertainty, which entails training a deep ensemble (epistemic) with nonnegative log-likelihood (aleatoric) loss in addition to the conventional applied losses terms. The proposed procedure can be paired with any DL reconstruction, enabling investigations of their predictive uncertainties on a pixel level. Five different architectures were investigated on the fastMRI database. The impact on the examined uncertainty of in-distributional and out-of-distributional data with changes to undersampling pattern, imaging contrast, imaging orientation, anatomy, and pathology were explored.

### Results
Predictive uncertainty could be captured and showed good correlation to normalized mean squared error. Uncertainty was primarily focused along the aliased anatomies and on hyperintense and hypointense regions. The proposed uncertainty measure was able to detect disease prevalence shifts. Distinct predictive uncertainty patterns were observed for changing network architectures.

### Conclusion
The proposed approach enables aleatoric and epistemic uncertainty prediction for DL-based MR reconstruction with an interpretable examination on a pixel level.


## Setup
You can use the provided script to install all packages and dependencies.
```
bash create_environment.sh
```

### Installed additional packages

- *submodule optox*: https://github.com/midas-tum/optox
- *submodule merlin*: https://github.com/midas-tum/merlin
- *submodule Medutils*: https://github.com/khammernik/medutils
- *submodule fastMRI*: https://github.com/facebookresearch/fastMRI
- *submodule fastMRI_data*: https://github.com/khammernik/sigmanet
- *submodule fastMRI-plus*: https://github.com/microsoft/fastmri-plus

## Structure
```
config/                 # Configuration files for all experiments and models
ensemble/dataset        # Data loader and transformations
ensemble/datasets       # Data set configs
ensemble/fastmri        # FastMRI data loader (from fastMRI submodule); no longer used
ensemble/models         # Model architectures
experiments/            # main scripts for training and evaluation
external/               # external code as submodules (see Setup)
utils/                  # utility functions for data filtering and experiment creation
```

## Usage

```
python3 experiments/train_ensemble.py --config <YAML_FILE> --experiment <EXPERIMENT_NAME> [--predict | --train]
```
YAML_FILE: Path to the configuration file in the config folder.<br/>
EXPERIMENT_NAME: Name of the experiment specified in the YAML_FILE. The experiment will be saved in the `config['result_dir']` folder.

## Cite

If you use the code in your project, please cite the paper:

```BibTeX
@article{https://doi.org/10.1002/mrm.30030,
author = {Küstner, Thomas and Hammernik, Kerstin and Rueckert, Daniel and Hepp, Tobias and Gatidis, Sergios},
title = {Predictive uncertainty in deep learning–based MR image reconstruction using deep ensembles: Evaluation on the fastMRI data set},
journal = {Magnetic Resonance in Medicine},
volume = {n/a},
number = {n/a},
pages = {},
keywords = {deep ensembles, deep learning, epistemic and aleatoric uncertainty, image reconstruction, MRI, uncertainty estimation},
doi = {https://doi.org/10.1002/mrm.30030},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.30030},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/mrm.30030},
}
```