import os
import sys
import joblib
from options.base_options import BaseOptions
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from utils.utils_model import choose_model, hyperparam_tune, load_variables, \
    split_data, tml_report, network_outer_report, network_report, extract_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, \
    accuracy_score, precision_score, recall_score
from utils.plot_utils import create_bar_plot, create_violin_plot, create_strip_plot, create_parity_plot
from math import sqrt
from data.rhcaa import rhcaa_diene
import argparse

from icecream import ic


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def train_tml_model_nested_cv(opt: argparse.Namespace, parent_dir:str) -> None:

    print('Initialising chiral ligands selectivity prediction using a traditional ML approach.')
    
    # Get the current working directory 
    current_dir = parent_dir

    # Load the data
    data = pd.read_csv(f'{opt.root}/raw/{opt.filename}')
    data = data[['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6',
               'ER7', 'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4', '%top', 'fold', 'index']]
    descriptors = ['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6',
               'ER7', 'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4']
    
    # Nested cross validation
    ncv_iterator = split_data(data)

    # Initiate the counter of the total runs and the total number of runs
    counter = 0
    TOT_RUNS = opt.folds*(opt.folds-1)    
    print("Number of splits: {}".format(opt.folds))
    print("Total number of runs: {}".format(TOT_RUNS))

    # Hyperparameter optimisation
    print("Hyperparameter optimisation starting...")
    X, y, _ = load_variables(f'{opt.root}/raw/learning_folds.csv')
    best_params = hyperparam_tune(X, y, choose_model(best_params=None, algorithm = opt.tml_algorithm), 123456789)
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
            model.fit(train_set[descriptors], train_set['%top'])
            # Predict the train set
            preds = model.predict(train_set[descriptors])
            train_rmse = sqrt(mean_squared_error(train_set['%top'], preds))
            # Predict the validation set
            preds = model.predict(val_set[descriptors])
            val_rmse = sqrt(mean_squared_error(val_set['%top'], preds))
            # Predict the test set
            preds = model.predict(test_set[descriptors])
            test_rmse = sqrt(mean_squared_error(test_set['%top'], preds))

            print('Outer: {} | Inner: {} | Run {}/{} | Train RMSE {:.3f} % | Val RMSE {:.3f} % | Test RMSE {:.3f} %'.\
                  format(outer, real_inner, counter, TOT_RUNS, train_rmse, val_rmse, test_rmse) )
            
            # Generate a report of the model performance
            tml_report(log_dir=f"{current_dir}/{opt.log_dir_results}/learning_set/results_{opt.tml_algorithm}/",
                       data = (train_set, val_set, test_set),
                       outer = outer,
                       inner = real_inner,
                       model = model,
                       )
            
            # Reset the variables of the training
            del model, train_set, val_set, test_set
        
        print('All runs for outer test fold {} completed'.format(outer))
        print('Generating outer report')

        # Generate a report of the model performance for the outer/test fold
        network_outer_report(
            log_dir=f"{current_dir}/{opt.log_dir_results}/learning_set/results_{opt.tml_algorithm}/Fold_{outer}_test_set/",
            outer=outer,
        )

        print('---------------------------------')
        
    print('All runs completed')



def predict_final_test_network(parent_dir:str, opt: argparse.Namespace) -> None:

    opt = BaseOptions().parse()

    current_dir = parent_dir
    
    # Load the final test set
    final_test =rhcaa_diene(opt, opt.filename_final_test, opt.mol_cols, opt.root_final_test, include_fold=False)
    test_loader = DataLoader(final_test, shuffle=False)

    # Load the data for tml
    test_set = pd.read_csv(f'{opt.root_final_test}/raw/{opt.filename_final_test}')

    experiments_gnn = os.path.join(current_dir, opt.log_dir_results, 'final_test', 'results_GNN')
    experiments_tml = os.path.join(current_dir, opt.log_dir_results, 'final_test', f'results_{opt.tml_algorithm}')

    for outer in range(1, opt.folds+1):
        print('Analysing models trained using as test set {}'.format(outer))
        for inner in range(1, opt.folds):
    
            real_inner = inner +1 if outer <= inner else inner
            
            print('Analysing models trained using as validation set {}'.format(real_inner))

            model_dir = os.path.join(current_dir, opt.log_dir_results, 'learning_set', 'results_GNN', f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            model = torch.load(model_dir+'/model.pth')
            model_params = torch.load(model_dir+'/model_params.pth')
            train_loader = torch.load(model_dir+'/train_loader.pth')
            val_loader = torch.load(model_dir+'/val_loader.pth')

            network_report(log_dir=experiments_gnn,
                           loaders=(train_loader, val_loader, test_loader),
                           outer=outer,
                           inner=real_inner,
                           loss_lists=[None, None, None],
                           model=model,
                           model_params=model_params,
                           best_epoch=None,
                           save_all=False)
            
            tml_dir = os.path.join(current_dir, opt.log_dir_results, 'learning_set', f'results_{opt.tml_algorithm}', f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            model = joblib.load(tml_dir+'/model.sav')
            train_data = pd.read_csv(tml_dir+'/train.csv')
            val_data = pd.read_csv(tml_dir+'/val.csv')

            tml_report(log_dir=experiments_tml,
                       outer=outer,
                       inner=real_inner,
                       model=model,
                       data=(train_data,val_data,test_set),
                       save_all=False,)
            
                        
        network_outer_report(log_dir=f"{experiments_gnn}/Fold_{outer}_test_set/", 
                             outer=outer)
        
        network_outer_report(log_dir=f"{experiments_tml}/Fold_{outer}_test_set/", 
                             outer=outer)


def plot_results(exp_dir, opt):

    experiments_gnn = os.path.join(exp_dir, 'results_GNN')
    experiments_tml = os.path.join(exp_dir, f'results_{opt.tml_algorithm}')

    r2_gnn, mae_gnn, rmse_gnn = [], [], []
    accuracy_gnn, precision_gnn, recall_gnn = [], [], []
    r2_tml, mae_tml, rmse_tml = [], [], []
    accuracy_tml, precision_tml, recall_tml = [], [], []

    results_all = pd.DataFrame(columns = ['index', 'Test_Fold', 'Val_Fold', 'Method', 'real_%top', 'predicted_%top'])
    
    for outer in range(1, opt.folds+1):

        outer_gnn = os.path.join(experiments_gnn, f'Fold_{outer}_test_set')
        outer_tml = os.path.join(experiments_tml, f'Fold_{outer}_test_set')

        metrics_gnn = extract_metrics(file=outer_gnn+f'/performance_outer_test_fold{outer}.txt')
        metrics_tml = extract_metrics(file=outer_tml+f'/performance_outer_test_fold{outer}.txt')

        r2_gnn.append(metrics_gnn['R2'])
        mae_gnn.append(metrics_gnn['MAE'])
        rmse_gnn.append(metrics_gnn['RMSE'])
        accuracy_gnn.append(metrics_gnn['Accuracy'])
        precision_gnn.append(metrics_gnn['Precision'])
        recall_gnn.append(metrics_gnn['Recall'])

        r2_tml.append(metrics_tml['R2'])
        mae_tml.append(metrics_tml['MAE'])
        rmse_tml.append(metrics_tml['RMSE'])
        accuracy_tml.append(metrics_tml['Accuracy'])
        precision_tml.append(metrics_tml['Precision'])
        recall_tml.append(metrics_tml['Recall'])

        for inner in range(1, opt.folds):
    
            real_inner = inner +1 if outer <= inner else inner
            
            gnn_dir = os.path.join(experiments_gnn, f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            df_gnn = pd.read_csv(gnn_dir+'/predictions_test_set.csv')
            df_gnn['Test_Fold'] = outer
            df_gnn['Val_Fold'] = real_inner
            df_gnn['Method'] = 'GNN'

            results_all = pd.concat([results_all, df_gnn], axis=0)

            tml_dir = os.path.join(experiments_tml, f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            df_tml = pd.read_csv(tml_dir+'/predictions_test_set.csv')
            df_tml['Test_Fold'] = outer
            df_tml['Val_Fold'] = real_inner
            df_tml['Method'] = opt.tml_algorithm

            results_all = pd.concat([results_all, df_tml], axis=0)

    save_dir = f'{exp_dir}/GNN_vs_{opt.tml_algorithm}'
    os.makedirs(save_dir, exist_ok=True)
    
    mae_mean_gnn = np.array([entry['mean'] for entry in mae_gnn])
    mae_gnn_std = np.array([entry['std'] for entry in mae_gnn])

    mae_mean_tml = np.array([entry['mean'] for entry in mae_tml])
    mae_tml_std = np.array([entry['std'] for entry in mae_tml])

    rmse_mean_gnn = np.array([entry['mean'] for entry in rmse_gnn])
    rmse_gnn_std = np.array([entry['std'] for entry in rmse_gnn])

    rmse_mean_tml = np.array([entry['mean'] for entry in rmse_tml])
    rmse_tml_std = np.array([entry['std'] for entry in rmse_tml])

    minimun = np.min(np.array([
        (mae_mean_gnn - mae_gnn_std).min(), 
        (mae_mean_tml - mae_tml_std).min(),
        (rmse_mean_gnn - rmse_gnn_std).min(), 
        (rmse_mean_tml - rmse_tml_std).min()]))
    
    maximun = np.max(np.array([
        (mae_mean_gnn + mae_gnn_std).max(),
        (mae_mean_tml + mae_tml_std).max(),
        (rmse_mean_gnn + rmse_gnn_std).max(), 
        (rmse_mean_tml + rmse_tml_std).max()]))
    
    create_bar_plot(means=(mae_mean_gnn, mae_mean_tml), stds=(mae_gnn_std, mae_tml_std), min = minimun, max = maximun, metric = 'MAE', save_path= save_dir, tml_algorithm=opt.tml_algorithm)
    create_bar_plot(means=(rmse_mean_gnn, rmse_mean_tml), stds=(rmse_gnn_std, rmse_tml_std), min = minimun, max = maximun, metric = 'RMSE', save_path= save_dir, tml_algorithm=opt.tml_algorithm)

    r2_mean_gnn = np.array([entry['mean'] for entry in r2_gnn])
    r2_gnn_std = np.array([entry['std'] for entry in r2_gnn])

    r2_mean_tml = np.array([entry['mean'] for entry in r2_tml])
    r2_tml_std = np.array([entry['std'] for entry in r2_tml])

    accuracy_mean_gnn = np.array([entry['mean'] for entry in accuracy_gnn])
    accuracy_gnn_std = np.array([entry['std'] for entry in accuracy_gnn])

    accuracy_mean_tml = np.array([entry['mean'] for entry in accuracy_tml])
    accuracy_tml_std = np.array([entry['std'] for entry in accuracy_tml])

    precision_mean_gnn = np.array([entry['mean'] for entry in precision_gnn])
    precision_gnn_std = np.array([entry['std'] for entry in precision_gnn])

    precision_mean_tml = np.array([entry['mean'] for entry in precision_tml])
    precision_tml_std = np.array([entry['std'] for entry in precision_tml])

    recall_mean_gnn = np.array([entry['mean'] for entry in recall_gnn])
    recall_gnn_std = np.array([entry['std'] for entry in recall_gnn])

    recall_mean_tml = np.array([entry['mean'] for entry in recall_tml])
    recall_tml_std = np.array([entry['std'] for entry in recall_tml])
    
    minimun = np.min(np.array([
    (r2_mean_gnn - r2_gnn_std).min(),
    (r2_mean_tml - r2_tml_std).min(),
    (accuracy_mean_gnn - accuracy_gnn_std).min(),
    (accuracy_mean_tml - accuracy_tml_std).min(),
    (precision_mean_gnn - precision_gnn_std).min(),
    (precision_mean_tml - precision_tml_std).min(),
    (recall_mean_gnn - recall_gnn_std).min(),
    (recall_mean_tml - recall_tml_std).min()
    ]))

    maximun = np.max(np.array([
    (r2_mean_gnn + r2_gnn_std).max(),
    (r2_mean_tml + r2_tml_std).max(),
    (accuracy_mean_gnn + accuracy_gnn_std).max(),
    (accuracy_mean_tml + accuracy_tml_std).max(),
    (precision_mean_gnn + precision_gnn_std).max(),
    (precision_mean_tml + precision_tml_std).max(),
    (recall_mean_gnn + recall_gnn_std).max(),
    (recall_mean_tml + recall_tml_std).max()
    ]))
    
    
    create_bar_plot(means=(r2_mean_gnn, r2_mean_tml), stds=(r2_gnn_std, r2_tml_std), min = minimun, max = maximun, metric = 'R2', save_path= save_dir, tml_algorithm=opt.tml_algorithm)
    create_bar_plot(means=(accuracy_mean_gnn, accuracy_mean_tml), stds=(accuracy_gnn_std, accuracy_tml_std), min = minimun, max = maximun, metric = 'Accuracy', save_path= save_dir, tml_algorithm=opt.tml_algorithm)
    create_bar_plot(means=(precision_mean_gnn, precision_mean_tml), stds=(precision_gnn_std, precision_tml_std), min = minimun, max = maximun, metric = 'Precision', save_path= save_dir, tml_algorithm=opt.tml_algorithm)
    create_bar_plot(means=(recall_mean_gnn, recall_mean_tml), stds=(recall_gnn_std, recall_tml_std), min = minimun, max = maximun, metric = 'Recall', save_path= save_dir, tml_algorithm=opt.tml_algorithm)

    results_all['Error'] = results_all['real_%top'] - results_all['predicted_%top']
    results_all['real_face'] = np.where(results_all['real_%top'] > 50, 1, 0)
    results_all['predicted_face'] = np.where(results_all['predicted_%top'] > 50, 1, 0)

    create_violin_plot(data=results_all, save_path= save_dir)
    create_strip_plot(data=results_all, save_path= save_dir)

    create_parity_plot(data=results_all, save_path= save_dir, tml_algorithm=opt.tml_algorithm)

    results_all = results_all.reset_index(drop=True)
    results_all.to_csv(f'{save_dir}/predictions_all.csv', index=False)


    print('All plots have been saved in the directory {}'.format(save_dir))

    gnn_predictions = results_all[results_all['Method'] == 'GNN']

    print('\n')

    print('Final metrics for GNN:')

    print('Accuracy: {:.3f}'.format(accuracy_score(gnn_predictions['real_face'], gnn_predictions['predicted_face'])))
    print('Precision: {:.3f}'.format(precision_score(gnn_predictions['real_face'], gnn_predictions['predicted_face'])))
    print('Recall: {:.3f}'.format(recall_score(gnn_predictions['real_face'], gnn_predictions['predicted_face'])))
    print('R2: {:.3f}'.format(r2_score(gnn_predictions['real_%top'], gnn_predictions['predicted_%top'])))
    print('MAE: {:.3f}'.format(mean_absolute_error(gnn_predictions['real_%top'], gnn_predictions['predicted_%top'])))
    print('RMSE: {:.3f} \n'.format(sqrt(mean_squared_error(gnn_predictions['real_%top'], gnn_predictions['predicted_%top']))))
    
    tml_predictions = results_all[results_all['Method'] == opt.tml_algorithm]
    print(f'Final metrics for {opt.tml_algorithm}:')
    print('Accuracy: {:.3f}'.format(accuracy_score(tml_predictions['real_face'], tml_predictions['predicted_face'])))
    print('Precision: {:.3f}'.format(precision_score(tml_predictions['real_face'], tml_predictions['predicted_face'])))
    print('Recall: {:.3f}'.format(recall_score(tml_predictions['real_face'], tml_predictions['predicted_face'])))
    print('R2: {:.3f}'.format(r2_score(tml_predictions['real_%top'], tml_predictions['predicted_%top'])))
    print('MAE: {:.3f}'.format(mean_absolute_error(tml_predictions['real_%top'], tml_predictions['predicted_%top'])))
    print('RMSE: {:.3f} \n'.format(sqrt(mean_squared_error(tml_predictions['real_%top'], tml_predictions['predicted_%top']))))
    

    

    





