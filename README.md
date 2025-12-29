
<img width="6180" height="1440" alt="AIForProdrugDesign" src="https://github.com/user-attachments/assets/ade6ada3-cc3a-4dd3-8ec6-08db3832d554" />

## Overview

Prodrugs are easily deployable chemical entities with beneficial pharmacokinetic properties; however, their rational design requires careful crafting of release mechanisms and holistic optimization of pharmacokinetic properties. Machine learning is poised to support rational design of prodrugs by efficiently filtering millions of generated designs down to the most promising candidates. Here, we designed and validated a novel machine learning pipeline for rapid and systematic design of prodrugs with desired properties. We also developed a subsampling approach for efficient application of our pair-wise [DeepDelta](https://github.com/RekerLab/DeepDelta) approach to larger datasets (>1500 datapoints). 

The associated publication is currently under review. 

We would like to thank the [Chemprop](https://github.com/chemprop/chemprop), [Llama](https://github.com/meta-llama/llama3), [Unsloth](https://github.com/unslothai/unsloth), [Molecule-RNN](https://github.com/shiwentao00/Molecule-RNN), [SmilesGPT](https://github.com/sanjaradylov/smiles-gpt/tree/master), [MolGan](https://github.com/nicola-decao/MolGAN),  [Scikit-learn](https://github.com/scikit-learn/scikit-learn), and [Chemical VAE](https://github.com/aspuru-guzik-group/chemical_vae) developers for making their code publicly available. 

<br />


## Descriptions of Folders

### DeepDelta_for_Prodrugs

Datasets, saved models, and code for applying DeepDelta for the two example prodrug case studies. Due to the large file size of the files for generated prodrugs and their predicted values, these results are stored on [Zenodo](https://zenodo.org/records/18079221): 10.5281/zenodo.18079221. 

### DeepDelta_Subsampling

Datasets and code for subsampling strategies for efficient application of the pair-wise DeepDelta approach to larger datasets (>1500 datapoints). Due to the large file size of results, these are stored on [Zenodo](https://zenodo.org/records/14894034): 10.5281/zenodo.14894034.

### Existing_Prodrug_Analysis

Datasets, code worksheet, and results for the analysis of currently approved and investigational prodrugs (Supplementary Figure 1).

### Generative_Models

Datasets and code worksheets for the generative models based on [Molecule-RNN](https://github.com/shiwentao00/Molecule-RNN) for the GRU and LSTM, [SmilesGPT](https://github.com/sanjaradylov/smiles-gpt/tree/master) for the GPT, [MolGan](https://github.com/nicola-decao/MolGAN) for the GAN, and [Scikit-learn](https://github.com/scikit-learn/scikit-learn) and [Chemical VAE](https://github.com/aspuru-guzik-group/chemical_vae) for the VAE. Please refer to the [Unsloth](https://github.com/unslothai/unsloth) implementation of [Llama 3.1](https://github.com/meta-llama/llama3) for the LLM.

### Generative_Prodrug_Analysis

Datasets, code worksheet, and results for the analysis of novel prodrugs designed using generative models (Figure 1).

<br />



## License

The copyrights of the software are owned by Duke University. As such, two licenses for this software are offered:
1. An open-source license under the GPLv2 license for non-commercial academic use.
2. A custom license with Duke University, for commercial use or uses without the GPLv2 license restrictions. 
