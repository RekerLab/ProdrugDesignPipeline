# Imports
import sys, os
import yaml

import deepchem as dc
import pandas as pd
import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose

import selfies as sf
from rdkit import Chem
import rdkit, rdkit.Chem, rdkit.Chem.Draw
from rdkit.Chem.Draw import IPythonConsole

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

################################################################################
## RNN-based Generation of Promoieties Directly onto Starting Drug Structures ##
################################################################################

# Adjustable variables
model_type = 'GRU' # Can be either GRU or LSTM
out_file_name = 'GRU_generated_excipients'
batch_size = 2048 # Default 2048, input number 1 or greater
num_batches = 20 # Default 20, input number 1 or greater
dataset_dir = './Molecule-RNN-main/dataset/PurchasableFragments.smi' # PurchasableFragments is the training data used for promoiety generation
vocab_path = './Molecule-RNN-main/vocab/selfies_merged_vocab.yaml' # selfies_merged_vocab is the associated vocabulary used for promoiety generation

with open('Molecule-RNN-main/train.yaml') as f: # Possibly adjust pathway based on folder location
  list_doc = yaml.safe_load(f)
  list_doc['rnn_config']['rnn_type'] = model_type
  list_doc['out_file_name'] = out_file_name
  list_doc['batch_size'] = batch_size
  list_doc['num_batches'] = num_batches
  list_doc['dataset_dir'] = dataset_dir
  list_doc['vocab_path'] = vocab_path
  with open('Molecule-RNN-main/train.yaml', "w") as f: # Possibly adjust pathway based on folder location
    yaml.dump(list_doc, f)
    
# Enter your API (in SMILES code) here to build directly off of this structure. Please press enter when done typing.
# Note: The last character of the SMILES string is where the promoiety will begin
# For cefuroxime, the correctly oriented SMILES string is on the next line:
# CO/N=C(/C1=CC=CO1)\C(=O)N[C@H]2[C@@H]3N(C2=O)C(=C(CS3)COC(=O)N)C(=O)O
# For A-1331852 the correctly oriented SMILES string with explicit hydrogens is on the next line:
# O=C(C1=NC(N2C([H])(C3=C(C([H])=C(C([H])=C3C(NC4=NC5=C(C([H])=C(C([H])=C5S4)[H])[H])=O)[H])C([H])(C2([H])[H])[H])[H])=C(C([H])=C1C6=C([H])N(C([H])(C78C([H])(C([H])(C([H])(C9(C8([H])[H])[H])[H])C([H])(C([H])(C9([H])[H])C7([H])[H])[H])[H])[H])N=C6[H])[H])O
# These explicit hydrogens prevent ring formations with the promoiety and original drug structure

with open('/content/Molecule-RNN-main/train.yaml') as f:
    list_doc = yaml.safe_load(f)
    while True:
      try:
        drug = sf.encoder(input('Please enter your DRUG in SMILES format: ').replace('\\\\','\\'))
        drug = drug.replace('\\','*')
        list_doc['drug'] = drug.replace('\\','*')
        print('Your encoded drug is: ' + str(drug))
        with open('/content/Molecule-RNN-main/train.yaml', "w") as f:
          yaml.dump(list_doc, f)
        print('Success!')
        break
      except:
        print('This is not a valid SMILES code; try again.')
        continue
        

# Model Training
!python ./Molecule-RNN-main/train.py

# Generate prodrugs
# Output file is located in the results folder with provided name (/content/Molecule-RNN-main/results/[out_file_name])
!python ./Molecule-RNN-main/sample.py