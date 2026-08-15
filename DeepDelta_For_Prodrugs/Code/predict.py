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
from sklearn import metrics
from scipy import stats as stats
from sklearn.model_selection import KFold
import chemprop

# Define abstract class to define interface of models
class abstractDeltaModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fit(self, x, y):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass    

class DeepDelta(abstractDeltaModel):
    epochs = None
    dirpath = None 

    def __init__(self, epochs=5, dirpath = None):
        self.epochs = epochs
        self.dirpath = dirpath


    def fit(self, x, y, prop, metric='r2'):
        
        self.dirpath = "{}_trained_model".format(prop)
        
        # create pairs of training data
        train = pd.merge(x, x, how='cross') 
        y_values = pd.merge(y, y, how='cross')
        train["Y"] = y_values.Y_y - y_values.Y_x
        del y_values 

        temp_datafile = tempfile.NamedTemporaryFile() 
        train.to_csv(temp_datafile.name, index=False)
        
        # store default arguments for ChemProp model
        arguments = [ 
            '--data_path', temp_datafile.name,
            '--separate_val_path', temp_datafile.name, 
            '--dataset_type', 'regression', 
            '--save_dir', self.dirpath,
            '--num_folds', '1',
            '--split_sizes', '1.0', '0', '0',
            '--ensemble_size', '1', 
            '--epochs', str(self.epochs),
            '--metric', metric,
            '--number_of_molecules', '2',
            '--aggregation', 'sum'
        ]
        
        args = chemprop.args.TrainArgs().parse_args(arguments)
        chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training) # Train

        temp_datafile.close()


    def predict(self, x, prop):

        dataset = pd.merge(x, x, how='cross') # Make pairs by cross-merging

        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name, 
            '--checkpoint_dir', "{}_trained_model".format(prop),
            '--number_of_molecules', '2'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args) # Predict

        predictions = pd.read_csv(temp_predfile.name)['Y']

        temp_datafile.close()
        temp_predfile.close()

        return predictions
    

    def predict_as_is(self, pred_df, prop):

        temp_datafile = tempfile.NamedTemporaryFile()
        pred_df.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name, 
            '--checkpoint_dir', "{}_trained_model".format(prop),
            '--number_of_molecules', '2'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args) # Predict

        predictions = pd.read_csv(temp_predfile.name)['Y']

        temp_datafile.close()
        temp_predfile.close()

        return predictions

    
    def __str__(self):
        return "DeepDelta" + str(self.epochs)

###########################
## FDA-approved Prodrugs ##
###########################

# Lipophilicity Testing
model = DeepDelta()
prop = 'Lipophilicity_no_prodrugs'
external_dataset = 'LipoFDAApproved'

pred_dataset = pd.read_csv('../Datasets/{}.csv'.format(external_dataset))
pred_x = pred_dataset[['SMILES_A', 'SMILES_B']] 
predictions = model.predict_as_is(pred_x, prop)
pred_dataset['Pred_Y'] = predictions
pred_dataset.to_csv('{}_{}.csv'.format (external_dataset, prop), index=False)
        
# Solubility Testing
model = DeepDelta()
prop = 'AqSol_no_prodrugs'
external_dataset = 'SolFDAApproved'

pred_dataset = pd.read_csv('../Datasets/{}.csv'.format(external_dataset))
pred_x = pred_dataset[['SMILES_A', 'SMILES_B']] 
predictions = model.predict_as_is(pred_x, prop)
pred_dataset['Pred_Y'] = predictions
pred_dataset.to_csv('{}_{}.csv'.format (external_dataset, prop), index=False)
        
        
        
#########################
## Cefuroxime Prodrugs ##
#########################

external_datasets = ['Cefuroxime_Generative_Products_Example']
properties = ['Caco2', 'B_Theta_MIC90']
API = 'CO/N=C(/C1=CC=CO1)\C(=O)N[C@H]2[C@@H]3N(C2=O)C(=C(CS3)COC(=O)N)C(=O)O'
model = DeepDelta()

for external_dataset in external_datasets:
    for prop in properties:
        pred_dataset = pd.read_csv('../Datasets/{}.csv'.format(external_dataset)) 
        pred_dataset['API'] = API
        pred_x = pred_dataset[['API', 'SMILES']]
        predictions = model.predict_as_is(pred_x, prop)
        pred_dataset['Pred_Y'] = predictions
        pred_dataset.to_csv('{}_{}.csv'.format (external_dataset, prop), index=False)



########################
## A-1331852 Prodrugs ##
########################

external_datasets = ['A-1331852_Generative_Products_Example']
properties = ['BCL_X_pIC50', 'Lipophilicity']
API = 'O=C(O)C1=NC(N2CC3=C(CC2)C=CC=C3C(NC4=NC5=CC=CC=C5S4)=O)=CC=C1C6=C(N(N=C6)CC78CC(CC(C8)C9)CC9C7)C'
model = DeepDelta()

for external_dataset in external_datasets:
    for prop in properties:
        pred_dataset = pd.read_csv('../Datasets/{}.csv'.format(external_dataset)) 
        pred_dataset['API'] = API
        pred_x = pred_dataset[['API', 'SMILES']]
        predictions = model.predict_as_is(pred_x, prop)
        pred_dataset['Pred_Y'] = predictions
        pred_dataset.to_csv('{}_{}.csv'.format (external_dataset, prop), index=False)
