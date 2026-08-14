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
from sklearn.ensemble import RandomForestRegressor as RF
from xgboost import XGBRegressor
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


    def fit(self, x, y, metric='r2'):
        
        self.dirpath = tempfile.NamedTemporaryFile().name # use temporary file to store model
        
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


    def predict(self, x): # For predicitions where you want to cross-merge datapoints

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
        return "DeepDelta" + str(self.epochs)




class DeepDelta_SubsampleSimilar1(abstractDeltaModel):
    epochs = None
    dirpath = None 

    def __init__(self, epochs=5, dirpath = None):
        self.epochs = epochs
        self.dirpath = dirpath


    def fit(self, x, y, metric='r2'):
        
        self.dirpath = tempfile.NamedTemporaryFile().name # use temporary file to store model
        
        # create pairs of training data - pair each with 50 other random datapoints
        train = pd.DataFrame()
        y_values = pd.DataFrame()

        x.reset_index(drop=True,inplace = True)
        y.reset_index(drop=True,inplace = True)

        
        for i in range(len(x)):
            x_current = pd.DataFrame([x[i]])
            x_other = pd.concat([x[:i], x[i+1:]])
            x_other.reset_index(drop=True,inplace = True)
            y_current = pd.DataFrame([y[i]])
            y_other = pd.concat([y[:i], y[i+1:]])
            y_other.reset_index(drop=True,inplace = True)

            # Prepare current datapoint's fingerprint
            x_current_mol = Chem.MolFromSmiles(x_current[0][0])
            x_current_fp = AllChem.GetMorganFingerprintAsBitVect(x_current_mol, 2, 1024)

            # Prepare Fingerprints of other datapoints
            mols = [Chem.MolFromSmiles(s) for s in x_other]
            fps_list = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols]

            # Get datapoint with maximum similarity
            most_similar_index = np.argmax(DataStructs.BulkTanimotoSimilarity(x_current_fp, fps_list))
            x_most_similar = pd.DataFrame([x_other[most_similar_index]])
            y_most_similar = pd.DataFrame([y_other[most_similar_index]])

            # Merge together
            x_merged = pd.merge(x_current, x_most_similar, how='cross')
            y_merged = pd.merge(y_current, y_most_similar, how='cross')

            train = pd.concat([train, x_merged])
            y_values = pd.concat([y_values, y_merged])

        train = train.rename(columns={'0_x': "SMILES_x", "0_y": "SMILES_y"})
        y_values = y_values.rename(columns={'0_x': "Y_x", "0_y": "Y_y"})
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
        return "DeepDelta" + str(self.epochs) + "_SubsampleSimilar1"
        
        
        

class DeepDelta_Subsample50(abstractDeltaModel):
    epochs = None
    dirpath = None 

    def __init__(self, epochs=5, dirpath = None):
        self.epochs = epochs
        self.dirpath = dirpath


    def fit(self, x, y, metric='r2'):
        
        self.dirpath = tempfile.NamedTemporaryFile().name # use temporary file to store model
        
        # create pairs of training data - pair each with 50 other random datapoints
        train = pd.DataFrame()
        y_values = pd.DataFrame()

        x.reset_index(drop=True,inplace = True)
        y.reset_index(drop=True,inplace = True)

        for i in range(len(x)):
            x_current = pd.DataFrame([x[i]])
            x_other = pd.concat([x[:i], x[i+1:]])
            y_current = pd.DataFrame([y[i]])
            y_other = pd.concat([y[:i], y[i+1:]])

            # Subsample a random 50 datapoints and a different random subsample each time
            x_other_selected = x_other.sample(n=50, random_state=i)
            y_other_selected = y_other.sample(n=50, random_state=i)

            # Merge with the 50 random datapoints into a new dataframe
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
        return "DeepDelta" + str(self.epochs) + "_Subsample50"
        
        

class DeepDelta_SubsampleOursSimilar(abstractDeltaModel):
    epochs = None
    dirpath = None 

    def __init__(self, epochs=5, dirpath = None):
        self.epochs = epochs
        self.dirpath = dirpath


    def fit(self, x, y, metric='r2'):
        
        self.dirpath = tempfile.NamedTemporaryFile().name # use temporary file to store model
        
        # create pairs of training data - pair each with appropriate number of other similar datapoints
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
            x_other.reset_index(drop=True,inplace = True)
            y_current = pd.DataFrame([y[i]])
            y_other = pd.concat([y[:i], y[i+1:]])
            y_other.reset_index(drop=True,inplace = True)

            # Prepare current datapoint's fingerprint
            x_current_mol = Chem.MolFromSmiles(x_current[0][0])
            x_current_fp = AllChem.GetMorganFingerprintAsBitVect(x_current_mol, 2, 1024)

            # Prepare Fingerprints of other datapoints
            mols = [Chem.MolFromSmiles(s) for s in x_other]
            fps_list = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols]

            # Get indexes datapoints with maximum similarity
            similarities = np.array(DataStructs.BulkTanimotoSimilarity(x_current_fp, fps_list))
            similarity_indexes = (-similarities).argsort()[:num_pairs]

            # Merge with the random datapoints into a new dataframe
            x_merged = pd.merge(x_current, x[similarity_indexes], how='cross')
            y_merged = pd.merge(y_current, y[similarity_indexes], how='cross')

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
        return "DeepDelta" + str(self.epochs) + "_OursSimilar"
        
        


class DeepDelta_SubsampleOursRandom(abstractDeltaModel):
    epochs = None
    dirpath = None 

    def __init__(self, epochs=5, dirpath = None):
        self.epochs = epochs
        self.dirpath = dirpath


    def fit(self, x, y, metric='r2'):
        
        self.dirpath = tempfile.NamedTemporaryFile().name # use temporary file to store model
        
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


class Trad_ChemProp(abstractDeltaModel):
    epochs = None
    dirpath = None  
    dirpath_single = None

    def __init__(self, epochs=50, dirpath = None, dirpath_single = None):
        self.epochs = epochs
        self.dirpath = dirpath
        self.dirpath_single = dirpath_single

    def fit(self, x, y, metric='r2'):
        self.dirpath_single = tempfile.NamedTemporaryFile().name # use temporary file to store model
        
        train = pd.DataFrame(np.transpose(np.vstack([x,y])),columns=["X","Y"])

        temp_datafile = tempfile.NamedTemporaryFile()
        train.to_csv(temp_datafile.name, index=False)

        # store default arguments for ChemProp model
        arguments = [
            '--data_path', temp_datafile.name,
            '--separate_val_path', temp_datafile.name,
            '--dataset_type', 'regression',
            '--save_dir', self.dirpath_single,
            '--num_folds', '1',
            '--split_sizes', '1.0', '0', '0',
            '--ensemble_size', '1',
            '--epochs', str(self.epochs),
            '--number_of_molecules', '1',
            '--metric', metric, 
            '--aggregation', 'sum'
        ]

        args = chemprop.args.TrainArgs().parse_args(arguments)
        chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training) # Train

        temp_datafile.close()


    def predict(self, x): 

        dataset = pd.DataFrame(x)
        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name, 
            '--checkpoint_dir', self.dirpath_single,
            '--number_of_molecules', '1'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args) # Make prediction

        predictions = pd.read_csv(temp_predfile.name)['Y'] 

        preds = pd.merge(predictions,predictions,how='cross') # Cross merge to make pairs

        temp_datafile.close()
        temp_predfile.close()

        return preds.Y_y - preds.Y_x   # Calculate and return the delta values
    
    def __str__(self):
        return "ChemProp" + str(self.epochs)




class Trad_RF(abstractDeltaModel):
    model = None

    def __init__(self):
        self.model = RF()

    def fit(self, x, y, metric='r2'):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        self.model.fit(fps,y) # Fit using traditional methods

    def predict(self, x):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        
        predictions = pd.DataFrame(self.model.predict(fps)) # Predict using traditional methods
        results = pd.merge(predictions,predictions,how='cross') # Cross merge into pairs after predictions
        return results['0_y'] - results['0_x']  # Calculate and return the delta values
    
    def __str__(self):
        return "RandomForest"

class Trad_XGBoost(abstractDeltaModel):
    model = None

    def __init__(self):
        self.model = XGBRegressor(tree_method='gpu_hist')

    def fit(self, x, y, metric='r2'):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        self.model.fit(fps,y) # Fit using traditional methods

    def predict(self, x):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        
        predictions = pd.DataFrame(self.model.predict(fps)) # Predict using traditional methods
        results = pd.merge(predictions,predictions,how='cross') # Cross merge into pairs after predictions
        return results['0_y'] - results['0_x']  # Calculate and return the delta values
    
    def __str__(self):
        return "XGBoost"
