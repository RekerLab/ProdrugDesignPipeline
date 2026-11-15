
<img width="6180" height="1440" alt="AIForProdrugDesign" src="https://github.com/user-attachments/assets/ade6ada3-cc3a-4dd3-8ec6-08db3832d554" />

## Overview

Prodrugs are easily deployable chemical entities with beneficial pharmacokinetic properties; however, their rational design requires careful crafting of release mechanisms and holistic optimization of pharmacokinetic properties. Machine learning is poised to support rational design of prodrugs by efficiently filtering millions of generated designs down to the most promising candidates. Here, we designed and validated a novel machine learning pipeline for rapid and systematic design of prodrugs with desired properties. We also developed a subsampling approach for efficient application of our pair-wise [DeepDelta](https://github.com/RekerLab/DeepDelta) approach to larger datasets (>1500 datapoints) described and available here. 

The associated publication is currently under review. 

We would like to thank the [Chemprop](https://github.com/chemprop/chemprop), [Llama](https://github.com/meta-llama/llama3), [Unsloth](https://github.com/unslothai/unsloth), [Molecule-RNN](https://github.com/shiwentao00/Molecule-RNN), [SmilesGPT](https://github.com/sanjaradylov/smiles-gpt/tree/master), [MolGAN](https://github.com/nicola-decao/MolGAN),  [Scikit-learn](https://github.com/scikit-learn/scikit-learn), and [Chemical VAE](https://github.com/aspuru-guzik-group/chemical_vae) developers for making their code publicly available. 

<br />

## Descriptions of Folders

### DeepDelta_for_Prodrugs

Datasets, saved models, and code for performance validation with FDA-approved prodrugs and the two proof-of-concept case studies. Due to the large file size for generated prodrugs and their predicted values, these results are stored on [Zenodo](https://zenodo.org/records/18079221): 10.5281/zenodo.18079221. 

### DeepDelta_Subsampling

Datasets and code for subsampling strategies for efficient application of the pair-wise DeepDelta approach to larger datasets (>1500 datapoints). Due to the large file size of results, these are stored on [Zenodo](https://zenodo.org/records/14894034): 10.5281/zenodo.14894034.

### Demo

Code to run a small scale version of predicting prodrug properties of designed prodrugs. 

### EnsembleSiameseProdrug

Submodule link to metabolic model GitHub page. 

### Existing_Prodrug_Analysis

Datasets, code, and results for the analysis of currently approved and investigational prodrugs (Figure S1).

### Generative_Models

Datasets and code for generative models to directly build promoieties onto drug structures (based on [Molecule-RNN](https://github.com/shiwentao00/Molecule-RNN)).

### Generative_Prodrug_Analysis

Datasets, code, and results for the analysis of novel prodrugs designed using generative models (Figure 1).

<br />

## System Requirements
* [RDKit](https://www.rdkit.org/docs/Install.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [numpy](https://numpy.org/)
* [pandas](https://github.com/pandas-dev/pandas)
* [Chemprop v1.5.2](https://github.com/chemprop/chemprop)

Given the larger size of delta datasets, we recommend using a GPU for significantly faster training.
To use Chemprop with GPUs, you will need:
* cuda >= 8.0
* cuDNN

Predictive models were implemented with the following dependencies:

Channels:
  - pytorch
  - conda-forge

Dependencies:
  - python>=3.7
  - flask>=1.1.2
  - gunicorn>=20.0.4
  - hyperopt>=0.2.3
  - matplotlib>=3.1.3
  - numpy>=1.18.1
  - pandas>=1.0.3
  - pandas-flavor>=0.2.0
  - pip>=20.0.2
  - pytorch>=1.4.0
  - rdkit>=2020.03.1.0
  - scikit-learn>=0.22.2.post1
  - scipy>=1.4.1
  - tensorboardX>=2.0
  - torchvision>=0.5.0
  - tqdm>=4.45.0
  - pip:
    - typed-argument-parser>=1.6.1
    - git+https://github.com/bp-kelley/descriptastorus

<br />

## Installation

```
git clone https://github.com/RekerLab/ProdrugDesignPipeline
```

An installation including all system requirements should take on the order of minutes for a "normal" desktop computer. 

<br />

## Demo

A small dataset of the synthesized cefuroxime prodrugs is used to quickly demonstrate how to predict properties of designed prodrugs. To do so, run the python file within the code folder. Provided in the results folder, the expected result is two csv files, one for each property of interest, with the predicted values in an additional column ('Y'). This should take on the order of minutes for a "normal" desktop computer. 

<br />

## Instructions for Use

Within each described folder, there are code, dataset, and results folders (unless the results are too larger and stored on Zenodo). To reproduce, run the code within the code folder. It will draw upon the folders in the dataset folder. The results are provided for reference. 

<br />

## License

The copyrights of the software are owned by Duke University. As such, two licenses for this software are offered:
1. An open-source license under the GPLv2 license for non-commercial academic use.
2. A custom license with Duke University, for commercial use or uses without the GPLv2 license restrictions. 



