import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import joblib
import os
import argparse
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from math import sqrt

import sys

sys.path.insert(0, 
                './Utils')

from Utils.utils import predict_ap, test_summary_outer
from Utils.Utils_traditional_ML import model_report
from Utils.ploting_functions import *

PARSER = argparse.ArgumentParser(description="Prediction of %top variable for a given new test set.")

PARSER.add_argument("-d", "--dir", dest="d", type=str, default="final_test/traditional_ML",
                    help="Name of the directory where plots will be saved.")
PARSER.add_argument("-m", "--mod", dest="m", type=str, default="RhCAA/Traditional_ML/GradientBoosting",
                    help="Name of the directory where the models are stored.")
PARSER.add_argument("-t", "--test_set", type=str, dest="t", default="Data/final_test/raw",
                    help="Directory where test set graphs are stored.")

ARGS = PARSER.parse_args()



def main():

    pwd = os.getcwd()
    test_data = os.path.join(pwd, ARGS.t, 'final_test.csv')

    experiments_dir = os.path.join(pwd, ARGS.d)
    os.makedirs(experiments_dir, exist_ok=True)
    
    metrics_all = []

    for outer in range(1,11):

        metrics_out = []

        current_out_dir = os.path.join(experiments_dir, f'{outer}')
        os.mkdir(current_out_dir)

        for inner in range(1,10):

            model_dir = os.path.join(pwd, ARGS.m, f'{outer}', f'{outer}_{inner}_model')

            model = joblib.load(model_dir+'/model.sav')

            df = predict_ap(model=model_dir+'/model.sav', data= test_data)

            test_set = pd.read_csv(test_data)
            train_data = pd.read_csv(model_dir+'/train.csv')
            val_data = pd.read_csv(model_dir+'/val.csv')

            df['error'] = df["real_top"] - df["predicted_top"]

            #create_st_parity_plot(real=df['real_top'], predicted=df['predicted_top'], figure_name='parity_plot', save_path=inner_dir)

            model_report(model_name="{}_{}_model".format(outer, inner),
                         outer=str(outer),
                         model_path=experiments_dir, 
                         model=model,
                         data=(train_data,val_data,test_set),
                         training_time=None,
                         save_all=True,
                         )


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
