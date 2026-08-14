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


class DeepDelta_SubsampleOursRandom(abstractDeltaModel):
    epochs = None
    dirpath = None 

    def __init__(self, epochs=5, dirpath = None):
        self.epochs = epochs
        self.dirpath = dirpath


    def fit(self, x, y, prop, metric='r2'):
        
        self.dirpath = "{}_trained_model".format(prop)
        
        # create pairs of training data - pair each with random datapoints
        train = pd.DataFrame()
        y_values = pd.DataFrame()

        x.reset_index(drop=True,inplace = True)
        y.reset_index(drop=True,inplace = True)

        if len(x) > 1000: # Dataset is large enough for subsampling
            num_pairs = math.floor(1000000 / len(x))
        else: # Dataset is small enough to not need subsampling
            num_pairs = len(x) - 1

        for i in range(len(x)):
            x_current = pd.DataFrame([x[i]])
            x_other = pd.concat([x[:i], x[i+1:]])
            y_current = pd.DataFrame([y[i]])
            y_other = pd.concat([y[:i], y[i+1:]])

            # Subsample random datapoints and a different random subsample each time
            x_other_selected = x_other.sample(n=num_pairs, random_state=i)
            y_other_selected = y_other.sample(n=num_pairs, random_state=i)

            # Merge with the random datapoints into a new dataframe
            x_merged = pd.merge(x_current, x_other_selected, how='cross')
            y_merged = pd.merge(y_current, y_other_selected, how='cross')

            train = pd.concat([train, x_merged])
            y_values = pd.concat([y_values, y_merged])

        train = train.rename(columns={0: "SMILES_x", "SMILES": "SMILES_y"})
        y_values = y_values.rename(columns={0: "Y_x", "Y": "Y_y"})
        train["Y"] = y_values.Y_y - y_values.Y_x

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


    def predict(self, x):

        dataset = pd.merge(x, x, how='cross') # Make pairs by cross-merging

        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name, 
            '--checkpoint_dir', self.dirpath,
            '--number_of_molecules', '2'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args) # Predict

        predictions = pd.read_csv(temp_predfile.name)['Y']

        temp_datafile.close()
        temp_predfile.close()

        return predictions
    
    def predict_as_is(self, pred_df, prop): # For predictions where you want to predict on 2 columns of drugs

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
        return "DeepDelta" + str(self.epochs) + "_OursRandom"

# Smaller Datasets
properties = ['B_Theta_MIC90', 'BCL_X_pIC50', 'Caco2', 'Lipophilicity']

for prop in properties:
    dataset = '../Datasets{}.csv'.format(prop) # Training dataset

    # Fit model on entire training dataset
    df = pd.read_csv(dataset)
    x = df[df.columns[0]]
    y = df[df.columns[1]]
    if len(x) < 1500:
        model = DeepDelta()
    else:
        model = DeepDelta_SubsampleOursRandom()
    model.fit(x,y,prop) 


