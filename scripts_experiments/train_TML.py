import os
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from utils.utils_model import choose_model, hyperparam_tune, load_variables, \
    split_data, tml_report, network_outer_report, calculate_morgan_fingerprints
from options.base_options import BaseOptions


def train_tml_model_nested_cv(opt) -> None:

    print('Initialising chiral ligands selectivity prediction using a traditional ML approach.')

    # Load the data
    filename = opt.filename[:-4] + '_folds' + opt.filename[-4:]
    data = pd.read_csv(f'{opt.root}/raw/{filename}')

    if opt.descriptors == 'bespoke':
        data = data[['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6',
                'ER7', 'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4', 'temp', 'ddG', 'fold', 'index']]
        descriptors = ['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6',
                'ER7', 'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4', 'temp']
        
    elif opt.descriptors == 'morgan':
        data = data[opt.mol_cols + ['temp', 'ddG', 'fold', 'index']]
        fingerprints = calculate_morgan_fingerprints(df = data, smiles_cols=opt.mol_cols, variance_threshold=.01)
        data = data.drop(opt.mol_cols, axis=1)
        descriptors = ['temp'] + fingerprints.columns.tolist() 
        print(f'Using {len(descriptors)} fingerprints')
        data = pd.concat([data, fingerprints], axis = 1)
    
    elif opt.descriptors == 'circus_fp':
        data = data[opt.mol_cols + ['temp', 'ddG', 'fold', 'index']]
        fingerprints = pd.read_csv('data/datasets/circus_descriptors/diene_circus_descriptors.csv')
        data = data.drop(opt.mol_cols, axis=1)
        descriptors = ['temp'] + fingerprints.columns.tolist()
        print(f'Using {len(descriptors)} fingerprints')
        data = pd.merge(data, fingerprints, left_index=True, right_index=True)

    else:
        raise ValueError('Descriptors not recognised. Please choose between bespoke or morgan')

    
    # Nested cross validation
    ncv_iterator = split_data(data)

    # Initiate the counter of the total runs and the total number of runs
    counter = 0
    TOT_RUNS = opt.folds*(opt.folds-1)    
    print("Number of splits: {}".format(opt.folds))
    print("Total number of runs: {}".format(TOT_RUNS))

    # Hyperparameter optimisation
    print("Hyperparameter optimisation starting...")
    X, y, _ = load_variables(data=data, descriptors=descriptors+['ddG'])
    best_params = hyperparam_tune(X, y, choose_model(best_params=None, algorithm = opt.tml_algorithm), opt.global_seed)
    print('Hyperparameter optimisation has finalised')
    print("Training starting...")
    print("********************************")
    
    # Loop through the nested cross validation iterators
    # The outer loop is for the outer fold or test fold
    for outer in range(1, opt.folds+1):
        # The inner loop is for the inner fold or validation fold
        for inner in range(1, opt.folds):

            # Inner fold is incremented by 1 to avoid having same inner and outer fold number for logging purposes
            real_inner = inner +1 if outer <= inner else inner
            # Increment the counter
            counter += 1

            # Get the train, validation and test sets
            train_set, val_set, test_set = next(ncv_iterator)
            # Choose the model
            model = choose_model(best_params, opt.tml_algorithm)
            # Fit the model
            model.fit(train_set[descriptors], train_set['ddG'])
            # Predict the train set
            preds = model.predict(train_set[descriptors])
            train_rmse = sqrt(mean_squared_error(train_set['ddG'], preds))
            # Predict the validation set
            preds = model.predict(val_set[descriptors])
            val_rmse = sqrt(mean_squared_error(val_set['ddG'], preds))
            # Predict the test set
            preds = model.predict(test_set[descriptors])
            test_rmse = sqrt(mean_squared_error(test_set['ddG'], preds))

            print('Outer: {} | Inner: {} | Run {}/{} | Train RMSE {:.3f} kJ/mol | Val RMSE {:.3f} kJ/mol | Test RMSE {:.3f} kJ/mol'.\
                  format(outer, real_inner, counter, TOT_RUNS, train_rmse, val_rmse, test_rmse) )
            
            # Generate a report of the model performance
            tml_report(log_dir=f"{opt.log_dir_results}/{opt.filename[:-4]}/results_TML/{opt.tml_algorithm}/{opt.descriptors}/",
                       data = (train_set, val_set, test_set),
                       outer = outer,
                       inner = real_inner,
                       model = model,
                       descriptors=descriptors
                       )
            
            # Reset the variables of the training
            del model, train_set, val_set, test_set
        
        print('All runs for outer test fold {} completed'.format(outer))
        print('Generating outer report')

        # Generate a report of the model performance for the outer/test fold
        network_outer_report(
            log_dir=f"{opt.log_dir_results}/{opt.filename[:-4]}/results_TML/{opt.tml_algorithm}/{opt.descriptors}/Fold_{outer}_test_set/",
            outer=outer,
        )

        print('---------------------------------')
        
    print('All runs completed')


if __name__ == "__main__":
    opt = BaseOptions().parse()
    train_tml_model_nested_cv(opt)
