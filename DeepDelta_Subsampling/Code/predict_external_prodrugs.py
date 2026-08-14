# Imports
import os
import tempfile
import shutil
import abc
import pandas as pd
import numpy as np
import math
from rdkit import Chem
from rdkit.Chem import AllChem
import chemprop
from models import *

training_data = ['BCL-X-Regression-pIC50', 'Lipophilicity', 'Caco2', 'B-Theta-MIC90']
external_data = ['Cefuroxime_Prodrugs_from_Approved_Promoieties', 'A133_Prodrugs_from_Approved_Promoieties'] 
models = [DeepDelta_SubsampleOursRandom()] # Full list: [DeepDelta(), DeepDelta_SubsampleSimilar1(), DeepDelta_Subsample50(), DeepDelta_SubsampleOursSimilar(), DeepDelta_SubsampleOursRandom()]

for model in models:
    for train in training_data:
        for external in external_data:
            dataset = '../Datasets/Prodrug_Properties/{}.csv'.format(train) # For training
            pred_dataset = '../Datasets/Prodrug_Datasets/{}.csv'.format(external) # For prediction
            
            # Fit model on entire training dataset
            df = pd.read_csv(dataset)
            x = df[df.columns[0]]
            y = df[df.columns[1]]
            model.fit(x,y) 
            
            # Predict on prodrug test set (CSV organized such that it API (original drug) is column 1 
            # and prodrug (new drug) is column 2
            pred_x = pred_dataset[['API_SMILES', 'Prodrug_SMILES']] # Specific for the dataset notation
            predictions = model.predict_as_is(pred_x, prop)
            pred_dataset['Pred_Y'] = predictions
            pred_dataset.to_csv('{}_{}.csv'.format (external_dataset, prop), index=False)

