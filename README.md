
<img width="6180" height="1440" alt="AIForProdrugDesign" src="https://github.com/user-attachments/assets/ade6ada3-cc3a-4dd3-8ec6-08db3832d554" />

## Overview

Prodrugs are easily deployable chemical entities with beneficial pharmacokinetic properties; however, their rational design requires careful crafting of release mechanisms and holistic optimization of pharmacokinetic properties. Machine learning is poised to support rational design of prodrugs by efficiently filtering millions of generated designs down to the most promising candidates. Here, we designed and validated a novel machine learning pipeline for rapid and systematic design of prodrugs with desired properties. We also developed a subsampling approach for efficient application of our pair-wise [DeepDelta](https://github.com/RekerLab/DeepDelta) approach to larger datasets (>1500 datapoints). 

The associated publication is currently under review. 

We would like to thank the [Chemprop](https://github.com/chemprop/chemprop), [Llama](https://github.com/unslothai/unsloth), [Molecule-RNN](https://github.com/shiwentao00/Molecule-RNN), [SmilesGPT](https://github.com/sanjaradylov/smiles-gpt/tree/master), [MolGan](https://github.com/nicola-decao/MolGAN), and [Scikit-learn](https://github.com/scikit-learn/scikit-learn) developers for making their machine learning algorithms publicly available. 

<br />


## Descriptions of Folders

### DeepDelta_Subsampling

Datasets and code for subsampling strategies for efficient application of the pair-wise DeepDelta approach to larger datasets (>1500 datapoints). Due to the large file size of results, these are stored on [Zenodo](https://zenodo.org/records/14894034): 10.5281/zenodo.14894034.

### Exisiting_Prodrug_Analysis

Datasets, code worksheet, and results for the analysis of currently approved and investigational prodrugs (Supplementary Figure 1).

### Generative_Prodrug_Analysis

Datasets, code worksheet, and results for the analysis of novel prodrugs designed using generative models (Figure 1).

<br />




## License

The copyrights of the software are owned by Duke University. As such, two licenses for this software are offered:
1. An open-source license under the GPLv2 license for non-commercial academic use.
2. A custom license with Duke University, for commercial use or uses without the GPLv2 license restrictions. 
