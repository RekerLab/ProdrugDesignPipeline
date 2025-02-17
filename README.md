
![Pipeline](https://github.com/user-attachments/assets/a51ca38c-c742-48de-8f9d-36cacd721959)


## Overview

Prodrugs are easily deployable chemical entities with beneficial pharmacokinetic properties; however, their rational design requires careful crafting of release mechanisms and holistic optimization of pharmacokinetic properties. Machine learning is poised to support rational design of prodrugs by efficiently filtering millions of generated designs down to the most promising candidates. Here, we designed and validated a novel machine learning pipeline for rapid and systematic design of prodrugs with desired properties.  

The associated publication is currently under review. 

We would like to thank the Chemprop, XGBoost, Llama, MolGan, SmilesGPT, Molecule-RNN, and the Scikit-learn developers for making their machine learning algorithms publicly available. 

## Requirements
* [RDKit](https://www.rdkit.org/docs/Install.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [numpy](https://numpy.org/)
* [pandas](https://github.com/pandas-dev/pandas)

Machine Learning Models
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
* [Chemprop v1.5.2](https://github.com/chemprop/chemprop)
* [XGBoost](https://xgboost.readthedocs.io/en/stable/gpu/index.html)

Given the larger size of delta datasets, we recommend using a GPU for significantly faster training.

To use Chemprop with GPUs, you will need:
* cuda >= 8.0
* cuDNN

<br />


## Descriptions of Folders

### Exisiting_Prodrug_Analysis

Datasets, code worksheet, and results for the analysis of currently approved and investigational prodrugs (Supplementary Figure 1).

### Generative_Prodrug_Analysis

Datasets, code worksheet, and results for the analysis of novel prodrugs designed using generative models (Figure 1).

### DeepDelta_For_Large_Datasets

Datasets and code for subsampling strategies for efficient application of the pair-wise DeepDelta approach to larger datasets (>1500 datapoints).

<br />



## License

The copyrights of the software are owned by Duke University. As such, two licenses for this software are offered:
1. An open-source license under the GPLv2 license for non-commercial academic use.
2. A custom license with Duke University, for commercial use or uses without the GPLv2 license restrictions. 
