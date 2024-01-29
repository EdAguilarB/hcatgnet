import torch
import numpy as np
from torch_geometric.loader import DataLoader
import os
import argparse
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from math import sqrt


import sys

sys.path.insert(0, 
                './Utils')

from Utils.dataset import ChiralLigands_regr
from Utils.utils import *
from Utils.ploting_functions import *

PARSER = argparse.ArgumentParser(description="Prediction of %top variable for a given new test set.")

PARSER.add_argument("-d", "--dir", dest="d", type=str, default="bicyclic_test/GNN",
                    help="Name of the directory where plots will be saved.")
PARSER.add_argument("-m", "--mod", dest="m", type=str, default="RhCASA/GNN",
                    help="Name of the directory where the models are stored.")
PARSER.add_argument("-t", "--test_set", type=str, dest="t", default="Data/Bicyclic",
                    help="Directory where test set graphs are stored.")

ARGS = PARSER.parse_args()



def main():

    pwd = os.getcwd()
    test_set = os.path.join(pwd, ARGS.t)
    
    bicyclic = ChiralLigands_regr(root = test_set, filename= 'bicyclic.csv')
    print("Dataset type: ", type(bicyclic))
    print("Dataset length: ", bicyclic.len())
    print("Dataset node features: ", bicyclic.num_features)
    print("Dataset target: ", bicyclic.num_classes)
    print("Dataset sample: ", bicyclic[0])
    print("Sample  nodes: ", bicyclic[0].num_nodes)
    print("Sample  edges: ", bicyclic[0].num_edges, '\n')

    test_loader = DataLoader(bicyclic, batch_size=1, shuffle=False)

    experiments_dir = os.path.join(pwd, ARGS.d)
    os.makedirs(experiments_dir, exist_ok=True)

    
    metrics_all = []

    for outer in range(1,11):

        metrics_out = []

        current_out_dir = os.path.join(experiments_dir, f'{outer}')
        os.mkdir(current_out_dir)

        for inner in range(1,10):

            model_dir = os.path.join(pwd, ARGS.m, f'{outer}', f'{outer}_{inner}_model')

            df = predict(model_arch=model_dir+'/model_r.pth', model_params=model_dir+'/GNN_params_r.pth', data_loader=test_loader)
            df['error'] = df["real_top"] - df["predicted_top"]

            #create_st_parity_plot(real=df['real_top'], predicted=df['predicted_top'], figure_name='parity_plot', save_path=inner_dir)

            
            model = torch.load(model_dir+'/model_r.pth')
            model_params = torch.load(model_dir+'/GNN_params_r.pth')
            train_loader = torch.load(model_dir+'/train_loader.pth')
            val_loader = torch.load(model_dir+'/val_loader.pth')


            model_report(model_name="{}_{}_model".format(outer, inner),
                         outer = "{}".format(outer),
                         model_path=experiments_dir,
                         model=model,
                         model_params=model_params,
                         loaders=[train_loader, val_loader, test_loader],
                         loss_lists=[None, None, None],
                         best_epoch=None,
                         task = 'r',
                         save_all=True)


            r2 = r2_score(df['real_top'], df['predicted_top'])
            mae = mean_absolute_error(df['real_top'], df['predicted_top'])
            rmse = sqrt(mean_squared_error(df['real_top'], df['predicted_top']))
            MAPE = mean_absolute_percentage_error(df['real_top'], df['predicted_top'])
            MBE = np.mean(df['error'])
            error_std = np.std(df['error'])

            metrics_inner = [r2, mae, rmse, MAPE, MBE, error_std]

            metrics_out.append(metrics_inner)


        metrics_out = np.array(metrics_out)

        test_summary_outer(metrics=metrics_out,
                           save_dir=current_out_dir,
                           outer=outer)
        
        metrics_all.append(metrics_out)


    metrics_all = np.array(metrics_all)
    metrics_all = metrics_all.reshape(90, 6)
    metrics_mean = np.mean(metrics_all, axis=0)
    metrics_std = np.std(metrics_all, axis = 0)

    file1 = open("{}/summary.txt".format(experiments_dir), "w")

    file1.write('Summary of results reported as mean ± std:\n')
    file1.write("R2 score: {:.3f} ± {:.3f}\n".format(metrics_mean[0], metrics_std[0]))
    file1.write("Mean absolute error: {:.3f} ± {:.3f}\n".format(metrics_mean[1], metrics_std[1]))
    file1.write("Root mean squared error: {:.3f} ± {:.3f}\n".format(metrics_mean[2], metrics_std[2]))
    file1.write("Mean absolute percentage error: {:.3f} ± {:.3f}\n".format(metrics_mean[3], metrics_std[3]))
    file1.write("Mean bias error: {:.3f} ± {:.3f}\n".format(metrics_mean[4], metrics_std[4]))
    file1.write("Error std: {:.3f} ± {:.3f}\n".format(metrics_mean[5], metrics_std[5]))

    file1.close()

    return 'All results were saved in the directory {}'.format(experiments_dir)




















if __name__ == '__main__':
    main()