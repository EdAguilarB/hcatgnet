import os
import sys
import joblib
from options.base_options import BaseOptions
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer
import pandas as pd
import numpy as np
from utils.utils_model import choose_model, hyperparam_tune, load_variables, \
    split_data, tml_report, network_outer_report, network_report, extract_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, \
    accuracy_score, precision_score, recall_score
from utils.plot_utils import create_bar_plot, create_violin_plot, create_strip_plot, \
    create_parity_plot, plot_importances
from model.gcn import GCN_explain
from math import sqrt
from data.rhcaa import rhcaa_diene
import argparse
import random
from utils.other_utils import explain_dataset, visualize_score_features, \
    plot_molecule_importance, get_graph_by_idx

from icecream import ic


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def train_tml_model_nested_cv(opt: argparse.Namespace, parent_dir:str) -> None:

    print('Initialising chiral ligands selectivity prediction using a traditional ML approach.')
    
    # Get the current working directory 
    current_dir = parent_dir

    # Load the data
    filename = opt.filename[:-4] + '_folds' + opt.filename[-4:]
    data = pd.read_csv(f'{opt.root}/raw/{filename}')
    data = data[['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6',
               'ER7', 'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4', 'temp', 'ddG', 'fold', 'index']]
    descriptors = ['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6',
               'ER7', 'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4', 'temp']
    
    # Nested cross validation
    ncv_iterator = split_data(data)

    # Initiate the counter of the total runs and the total number of runs
    counter = 0
    TOT_RUNS = opt.folds*(opt.folds-1)    
    print("Number of splits: {}".format(opt.folds))
    print("Total number of runs: {}".format(TOT_RUNS))

    # Hyperparameter optimisation
    print("Hyperparameter optimisation starting...")
    X, y, _ = load_variables(f'{opt.root}/raw/{opt.filename}', descriptors=descriptors+['ddG'])
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

            print('Outer: {} | Inner: {} | Run {}/{} | Train RMSE {:.3f} % | Val RMSE {:.3f} % | Test RMSE {:.3f} %'.\
                  format(outer, real_inner, counter, TOT_RUNS, train_rmse, val_rmse, test_rmse) )
            
            # Generate a report of the model performance
            tml_report(log_dir=f"{current_dir}/{opt.log_dir_results}/learning_set/results_{opt.tml_algorithm}/",
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

    descriptors = ['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6',
               'ER7', 'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4', 'temp']

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
                       save_all=False,
                       descriptors=descriptors)
            
                        
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

    results_all = pd.DataFrame(columns = ['index', 'Test_Fold', 'Val_Fold', 'Method', 'real_ddG', 'predicted_ddG'])
    
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

            df_gnn = pd.read_csv(gnn_dir+'/predictions_test_set.csv', index_col=0)

            df_gnn['Test_Fold'] = outer
            df_gnn['Val_Fold'] = real_inner
            df_gnn['Method'] = 'GNN'

            results_all = pd.concat([results_all, df_gnn], axis=0)

            tml_dir = os.path.join(experiments_tml, f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            df_tml = pd.read_csv(tml_dir+'/predictions_test_set.csv', index_col=0)
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

    results_all['Error'] = results_all['real_ddG'] - results_all['predicted_ddG']
    results_all['real_face'] = np.where(results_all['real_ddG'] > 0, 1, 0)
    results_all['predicted_face'] = np.where(results_all['predicted_ddG'] > 0, 1, 0)

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
    print('R2: {:.3f}'.format(r2_score(gnn_predictions['real_ddG'], gnn_predictions['predicted_ddG'])))
    print('MAE: {:.3f}'.format(mean_absolute_error(gnn_predictions['real_ddG'], gnn_predictions['predicted_ddG'])))
    print('RMSE: {:.3f} \n'.format(sqrt(mean_squared_error(gnn_predictions['real_ddG'], gnn_predictions['predicted_ddG']))))
    
    tml_predictions = results_all[results_all['Method'] == opt.tml_algorithm]
    print(f'Final metrics for {opt.tml_algorithm}:')
    print('Accuracy: {:.3f}'.format(accuracy_score(tml_predictions['real_face'], tml_predictions['predicted_face'])))
    print('Precision: {:.3f}'.format(precision_score(tml_predictions['real_face'], tml_predictions['predicted_face'])))
    print('Recall: {:.3f}'.format(recall_score(tml_predictions['real_face'], tml_predictions['predicted_face'])))
    print('R2: {:.3f}'.format(r2_score(tml_predictions['real_ddG'], tml_predictions['predicted_ddG'])))
    print('MAE: {:.3f}'.format(mean_absolute_error(tml_predictions['real_ddG'], tml_predictions['predicted_ddG'])))
    print('RMSE: {:.3f} \n'.format(sqrt(mean_squared_error(tml_predictions['real_ddG'], tml_predictions['predicted_ddG']))))
    

def explain_model(exp_path:str, opt: argparse.Namespace) -> None:

    outer, inner = opt.explain_model[0], opt.explain_model[1]

    print('Analysing outer {}, inner {}'.format(outer, inner))

    model_path = os.path.join(exp_path, f'Fold_{outer}_test_set', f'Fold_{inner}_val_set')

    train_loader = torch.load(os.path.join(model_path, 'train_loader.pth'))
    val_loader = torch.load(os.path.join(model_path, 'val_loader.pth'))
    test_loader = torch.load(os.path.join(model_path, 'test_loader.pth'))

    all_data = train_loader.dataset + val_loader.dataset + test_loader.dataset

    loader_all = DataLoader(all_data)

    model = GCN_explain(opt, n_node_features=all_data[0].num_node_features)
    model_params = torch.load(os.path.join(model_path, 'model_params.pth'))
    model.load_state_dict(model_params)

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )

    loader = DataLoader(test_loader.dataset)


    ligand_masks, substrate_masks, boron_masks, all_masks  = explain_dataset(test_loader.dataset, 
                                                                             explainer)

    ligands = visualize_score_features(score = ligand_masks)
    ligands = ligands.loc[ligands['score'] != 0]
    ligands['labels'] = ligands['labels'].apply(lambda m: 'L. '+m)
    print('Ligand node features score: \n', ligands)

    substrate = visualize_score_features(score = substrate_masks)
    substrate = substrate.loc[substrate['score'] != 0]
    substrate['labels'] = substrate['labels'].apply(lambda m: 'S. '+m)
    print('Substrate node features score: \n', substrate)

    boron = visualize_score_features(score = boron_masks)
    boron = boron.loc[boron['score'] != 0]
    boron['labels'] = boron['labels'].apply(lambda m: 'BR. '+m)
    print('Boron node features score: \n', boron)

    df = pd.concat([ligands, substrate, boron])
    df = df.sort_values('score', ascending=False)
    df['score'] = df['score'].astype(int)

    plot_importances(df = df, save_path=os.path.join(exp_path, f'Fold_{outer}_test_set', f'Fold_{inner}_val_set'))


    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('ShapleyValueSampling'),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )

    loader = DataLoader(loader_all.dataset)

    for molecule in tqdm(opt.explain_reactions):
        mol = get_graph_by_idx(loader, molecule)
        print('Analysing reaction {}'.format(molecule))
        print('Ligand id: {}'.format(mol.ligand_id[0]))
        print('Reaction ddG: {:.2f}'.format(mol.y.item()))
        print('Reaction predicted ddG: {:.2f}'.format(explainer.get_prediction(x = mol.x, edge_index=mol.edge_index, batch_index=mol.batch).item()))
        explanation = explainer(x = mol.x, edge_index=mol.edge_index,  batch_index=mol.batch)
        plot_molecule_importance(mol_graph=mol, mol='l', explanation=explanation, palette='normal')


