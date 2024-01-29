import numpy as np
import pandas as pd
import os
import time
import argparse

import sys

sys.path.insert(0, 
                './Utils')

from Utils.Utils_traditional_ML import model_report, load_variables, hyperparam_tune, \
    choose_model, set_seed, split_data, calculate_metrics, outer_summary_report, experiment_summary_report


from icecream import ic

PARSER = argparse.ArgumentParser(description="Perform nested cross validation for GNN for prediction of internal energy of Molybdenum Carbides.")

PARSER.add_argument("-d", "--dir", dest="d", type=str, default="RhCASA/traditional_ML/",
                    help="Name of the directory where experiments will be saved.")
PARSER.add_argument("-r", "--random", type=int, dest="r", default=123456789,
                    help="Random seed used to run the experiment.")
PARSER.add_argument("-a", "--algorithm", type=str, dest="a", default='rf',
                    help="Traditional ML algorithm to use. Allowed values: lr for linear regression, gb for gradient boosting, or rf for random forest.")

ARGS = PARSER.parse_args()

HYPERPARAMS = {}
HYPERPARAMS['splits'] = 10


def main():

    print('Initialising chiral ligands selectivity prediction using a traditional ML approach.')

    current_dir = os.getcwd()
    parent_dir = os.path.join(current_dir, ARGS.d)

    os.makedirs(parent_dir, exist_ok=True)

    print("Saving the results in the parent directory: {}".format(parent_dir))

    TOT_RUNS = HYPERPARAMS["splits"] * (HYPERPARAMS["splits"]-1)

    print("Number of splits: {}".format(HYPERPARAMS["splits"]))
    print("Total number of runs: {}".format(TOT_RUNS))

    data = pd.read_csv(f'Data/Monocyclic/raw/monocyclic_folds.csv')


    data = data[['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6',
               'ER7', 'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4', '%top', 'fold', 'index']]
    
    descriptors = ['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6',
               'ER7', 'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4']
    

    print("Hyperparameter optimisation starting...")

    X, y, feat_names = load_variables(f'Data/Monocyclic/raw/monocyclic_folds.csv')

    best_params = hyperparam_tune(X, y, choose_model(best_params=None, algorithm = ARGS.a), set_seed(ARGS.r))

    print('Hyperparameter optimisation has finalised')
    

    ncv_iterator = split_data(data)
    counter = 1

    metrics_all_train, metrics_all_val, metrics_all_test = [], [], []

    print("Training starting...")
    print("********************************")

    t_initialisation = time.time()
    
    for outer in range(len(np.unique(data['fold']))):

        t_outer = time.time()

        os.mkdir("{}/{}".format(parent_dir, outer+1))

        metrics_train, metrics_val, metrics_test = [], [], []

        for inner in range(len(np.unique(data['fold'])) -1 ):

            t0 = time.time()

            train_set, val_set, test_set = next(ncv_iterator)

            model = choose_model(best_params, ARGS.a)

            model.fit(train_set[descriptors], train_set['%top'])

            preds = model.predict(train_set[descriptors])

            train_metrics = calculate_metrics(train_set['%top'], preds)
            metrics_train.append(train_metrics)

            preds = model.predict(val_set[descriptors])
            val_metrics = calculate_metrics(val_set['%top'], preds)
            metrics_val.append(val_metrics)

            preds = model.predict(test_set[descriptors])
            test_metrics = calculate_metrics(np.array(test_set['%top']), preds, all_metrics=True)
            metrics_test. append(test_metrics)

            print('Outer: {} | Inner: {} | Run {}/{} | Train RMSE {:.4f} % | Val RMSE {:.4f} % | Test RMSE {:.4f} %'.\
                  format(outer+1, inner+1, counter, TOT_RUNS, train_metrics[2], val_metrics[2], test_metrics[2]) )
            
            training_time_inner = (time.time() - t0)/60  

            model_report(model_name="{}_{}_model".format(outer+1, inner+1),
                         outer = outer+1, 
                         model_path="{}".format(parent_dir),
                         model = model,
                         data=(train_set, val_set, test_set),
                         training_time=training_time_inner)
            
            counter += 1

            del model, train_set, val_set, test_set
        

        metrics_train = np.array(metrics_train)
        metrics_val = np.array(metrics_val)
        metrics_test = np.array(metrics_test)

        training_time_outer = (time.time() - t_outer)/60  
            
        outer_summary_report(run_name='outer_summary_{}'.format(outer+1),
                            run_path='{}/{}'.format(parent_dir, outer+1),
                            best_metrics=(metrics_train, metrics_val, metrics_test),
                            training_time=training_time_outer)
        
        metrics_all_train.append(metrics_train)
        metrics_all_val.append(metrics_val)
        metrics_all_test.append(metrics_test)

    training_time_all = (time.time() - t_initialisation)/60  


    experiment_summary_report(experiment_name='summary_all_experiments',
                              run_path='{}'.format(parent_dir),
                              metrics=(metrics_all_train,metrics_all_val, metrics_all_test),
                              training_time=training_time_all)
    
    return 'Experiment has finalised.'


if __name__ == "__main__":  
    main()  