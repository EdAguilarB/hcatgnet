import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,\
    mean_absolute_percentage_error, accuracy_score, precision_score, recall_score, f1_score
from math import sqrt
from copy import copy, deepcopy
from datetime import date, datetime
import os
import pickle
from ploting_functions import *

from icecream import ic



def set_seed(seed = 123456789):

    np.random.seed(seed)
    print('\nGlobal seed: ', seed)

    return seed


def split_data(df:pd.DataFrame):
        
    for outer in np.unique(df['fold']):
        proxy = copy(df)
        test = proxy[proxy['fold'] == outer]

        for inner in np.unique(df.loc[df['fold'] != outer, 'fold']):

            val = proxy.loc[proxy['fold'] == inner]
            train = proxy.loc[(proxy['fold'] != outer) & (proxy['fold'] != inner)]
            yield deepcopy((train, val, test))


def calculate_metrics(y_true:list, y_predicted: list, all_metrics=False, task = 'r'):

    if task == 'r':

        r2 = r2_score(y_true=y_true, y_pred=y_predicted)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_predicted)
        rmse = sqrt(mean_squared_error(y_true=y_true, y_pred=y_predicted))  
        metrics = [r2, mae, rmse]

        if all_metrics is True:
            error = [(y_predicted[i]-y_true[i]) for i in range(len(y_true))]
            prctg_error = [ abs(error[i] / y_true[i]) for i in range(len(error))]
            mbe = np.mean(error)
            mape = np.mean(prctg_error)
            error_std = np.std(error)
            metrics += [mbe, mape, error_std]


    elif task == 'c':
        accuracy = accuracy_score(y_true=y_true, y_pred=y_predicted)
        precision = precision_score(y_true=y_true, y_pred=y_predicted)
        recall = recall_score(y_true=y_true, y_pred=y_predicted)
        f1 = f1_score(y_true=y_true, y_pred=y_predicted)

        metrics = [accuracy, precision, recall, f1]

    return np.array(metrics)



def pred_sets(model, data):

    descriptors = ['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6',
               'ER7', 'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4']
    y_pred = model.predict(data[descriptors])
    y_true = list(data['%top'])
    idx = list(data['index'])
    
    return np.array(y_pred), np.array(y_true), np.array(idx)



def model_report(model_name,
                 outer, 
                 model_path,
                 model,
                 data,
                 training_time,
                 save_all='True'):
    
    # 1) Get time of the run
    today = date.today()
    today_str = str(today.strftime("%d-%b-%Y"))
    time = str(datetime.now())[11:]
    time = time[:8]
    run_period = "{}, {}\n".format(today_str, time)

    # 2) Unfold  train/val/test dataloaders
    train_data, val_data, test_data = data[0], data[1], data[2]
    N_train, N_val, N_test = len(train_data), len(val_data), len(test_data)
    N_tot = N_train + N_val + N_test 

    #this creates a directory to store results if the directory does not exist
    log_dir = "{}/{}/{}".format(model_path, outer, model_name)
    os.mkdir(log_dir)



    # 9) Save dataframes for future use
    if save_all:
        train_data.to_csv("{}/train.csv".format(log_dir))
        val_data.to_csv("{}/val.csv".format(log_dir))

    test_data.to_csv("{}/test.csv".format(log_dir))

    if save_all:
        pickle.dump(model, open("{}/model.sav".format(log_dir), 'wb'))
   
    
# Performance Report
    file1 = open("{}/performance.txt".format(log_dir), "w")
    file1.write(run_period)
    file1.write("---------------------------------------------------------\n")
    file1.write("Traditional ML algorithm Performance\n")
    file1.write("Dataset Size = {}\n".format(N_tot))
    if training_time:
        file1.write("Training time = {} min\n".format(training_time))
    file1.write("***************\n")

    metrics = ['R2', 'MAE', 'RMSE', 'Mean Bias Error', \
                'Mean absolute percentage error', 'Error Standard Deviation']
    

    y_pred, y_true, idx = pred_sets(model=model, data=train_data)
    metric_vals = calculate_metrics(y_true=y_true, y_predicted=y_pred, all_metrics=True)

    file1.write("Training set\n")
    file1.write("Training set size = {}\n".format(N_train))

    for name, value in zip(metrics, metric_vals):
        file1.write("{} Train = {:.3f} \n".format(name, value))

    file1.write("***************\n")

    y_pred, y_true, idx = pred_sets(model=model, data=val_data)
    metric_vals = calculate_metrics(y_true=y_true, y_predicted=y_pred, all_metrics=True)

    file1.write("Validation set\n")
    file1.write("Validation set size = {}\n".format(N_val))

    for name, value in zip(metrics, metric_vals):
        file1.write("{} Validation = {:.3f} \n".format(name, value))

    file1.write("***************\n")

    y_pred, y_true, idx = pred_sets(model=model, data=test_data)

    side_pred = np.where(y_pred>50,1,0)
    side_true = np.where(y_true>50,1,0)

    metrics_class = calculate_metrics(y_true=side_true, y_predicted=side_pred, task='c')

    file1.write("Test set\n")
    file1.write("Test Set Total Size = {}\n".format(N_test))

    metrics_class_names = ['Accuracy', 'Precision',  'Recall', 'F1']

    for name, value in zip(metrics_class_names, metrics_class):
        file1.write("{} Test = {:.3f} \n".format(name, value))


    error = abs(y_true-y_pred)
    y_true = y_true[error<50]
    y_pred = y_pred[error<50]
    idx = idx[error<50]

    metric_vals = calculate_metrics(y_true=y_true, y_predicted=y_pred, all_metrics=True)


    file1.write("Test Set Total Correct Side  Predictions = {}\n".format(len(y_true)))


    for name, value in zip(metrics, metric_vals):
        file1.write("{} Test = {:.3f} \n".format(name, value))


    file1.write("---------------------------------------------------------\n")


    create_st_parity_plot(real=y_true, predicted=y_pred, figure_name=model_name, save_path=log_dir)
    create_it_parity_plot(y_true, y_pred, idx, '{}.html'.format(model_name), "{}".format(log_dir))

    file1.write("OUTLIERS (TEST SET)\n")


    error_test = [(y_pred[i] - y_true[i]) for i in range(len(y_true))] 
    abs_error_test = [abs(error_test[i]) for i in range(len(y_true))]
    std_error_test = np.std(error_test)

    outliers_list, outliers_error_list, index_list = [], [], []
    counter = 0

    for sample in range(len(y_true)):
        if abs_error_test[sample] >= 3 * std_error_test:  
            counter += 1
            outliers_list.append(idx[sample])
            outliers_error_list.append(error_test[sample])
            index_list.append(sample)
            if counter < 10:
                file1.write("0{}) {}    Error: {:.2f} %    (index={})\n".format(counter, idx[sample], error_test[sample], sample))
            else:
                file1.write("{}) {}    Error: {:.2f} %    (index={})\n".format(counter, idx[sample], error_test[sample], sample))

    file1.close()

    return "Model saved in {}".format(log_dir)


def outer_summary_report(run_name,
                          run_path,
                          best_metrics, 
                          training_time):
        
    set_names = ['Training set', 'Validation set', 'Test set']

    file1 = open("{}/{}.txt".format(run_path, run_name), "w")

    file1.write("Total training time = {} min\n".format(training_time))
        
    for set, name in zip(best_metrics, set_names):


        if set.shape[0] == 1:

            metric_row = set[0]

            file1.write("***************\n")
            file1.write("{}\n".format(name))
            file1.write("R2 = {:.3f} \n".format(float(metric_row[0])))
            file1.write("MAE = {:.3f} \n".format(metric_row[1]))
            file1.write("RMSE = {:.3f} \n".format(metric_row[2]))

            if len(metric_row) >3:
                file1.write("MBE = {:.3f} \n".format(metric_row[3]))
                file1.write("MAPE = {:.3f} \n".format(metric_row[4]))
                file1.write("Error std. = {:.3f} \n".format(metric_row[5]))

        else:
            mean_metrics = np.mean(set, axis = 0)
            std_metrics = np.std(set, axis = 0)

            file1.write("***************\n")
            file1.write("{}\n".format(name))
            file1.write("R2 = {:.3f} ± {:.3f} \n".format(mean_metrics[0],
                                                         std_metrics[0]))
            file1.write("MAE = {:.3f} ± {:.3f} \n".format(mean_metrics[1],
                                                          std_metrics[1]))
            file1.write("RMSE = {:.3f} ± {:.3f} \n".format(mean_metrics[2],
                                                           std_metrics[2]))

            if len(mean_metrics) > 3:
                file1.write("MBE = {:.3f} ± {:.3f} \n".format(mean_metrics[3],
                                                         std_metrics[3]))
                file1.write("MAPE = {:.3f} ± {:.3f} \n".format(mean_metrics[4],
                                                         std_metrics[4]))
                file1.write("Error std. = {:.3f} ± {:.3f} \n".format(mean_metrics[5],
                                                         std_metrics[5]))
    
    file1.close()

    return 'Outer {} summary file has been saved in {}/{}'.format(run_name, run_path, run_name)



def experiment_summary_report(experiment_name,
                              run_path,
                              metrics, 
                              training_time):
    

    
    set_names = ['Training set', 'Validation set', 'Test set']

    file1 = open("{}/{}.txt".format(run_path, experiment_name), "w")

    file1.write("---------------------------------------------------------\n")
    file1.write("MLR\n")
    file1.write("TRAINING PROCESS\n")
    file1.write("Total training time = {} min\n".format(training_time))
    #file1.write("Data Split (Train/Val/Test) = {}-{}-{} %\n".format(*split_percentage(hyperparams["splits"])))
    file1.write("---------------------------------------------------------\n")

    for set, name in zip(metrics, set_names):

        array = np.array(set)

        mean_metrics = np.mean(array, axis = 0)[0]
        std_metrics = np.std(array, axis = 0)[0]

        file1.write("***************\n")
        file1.write("{}\n".format(name))

        file1.write("R2 = {:.3f} ± {:.3f} \n".format(mean_metrics[0],
                                                     std_metrics[0]))
        file1.write("MAE = {:.3f} ± {:.3f} \n".format(mean_metrics[1],
                                                      std_metrics[1]))
        file1.write("RMSE = {:.3f} ± {:.3f} \n".format(mean_metrics[2],
                                                       std_metrics[2]))
        
        if len(mean_metrics) > 3:
            file1.write("MBE = {:.3f} ± {:.3f} \n".format(mean_metrics[3],
                                                     std_metrics[3]))
            file1.write("MAPE = {:.3f} ± {:.3f} \n".format(mean_metrics[4],
                                                     std_metrics[4]))
            file1.write("Error std. = {:.3f} ± {:.3f} \n".format(mean_metrics[5],
                                                     std_metrics[5]))
            
    file1.close()

    return '{} summary file has been saved in {}/{}.txt'.format(experiment_name, run_path, experiment_name)



def load_variables(path:str):

    descriptors = ['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6',
               'ER7', 'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4', '%top']

    data = pd.read_csv(path)

    data = data.filter(descriptors)

    #remove erroneous data
    data = data.dropna(axis=0)


    X = data.drop(['%top'], axis = 1)
    X = RobustScaler().fit_transform(np.array(X))
    y = data['%top']
    print('Features shape: ', X.shape)
    print('Y target variable shape: ' , y.shape)

    return X, y, descriptors



def choose_model(best_params, algorithm):

    if best_params == None:
        if algorithm == 'rf':
            return RandomForestRegressor()
        if algorithm == 'lr':
            return LinearRegression()
        if algorithm == 'gb':
            return GradientBoostingRegressor()

    else:
        if algorithm == 'rf':
            return RandomForestRegressor(n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], min_samples_leaf=best_params['min_samples_leaf'], 
                                     min_samples_split=best_params['min_samples_split'], random_state=best_params['random_state'])
        if algorithm == 'lr':
            return LinearRegression()
        if algorithm == 'gb':
            return GradientBoostingRegressor(loss = best_params['loss'], learning_rate=best_params['learning_rate'],n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"],
                                             min_samples_leaf=best_params['min_samples_leaf'], min_samples_split=best_params['min_samples_split'], random_state=best_params['random_state'])



def hyperparam_tune(X, y, model, seed):

    print('ML algorithm to be tunned:', str(model))


    if str(model) == 'LinearRegression()':
        return None
    
    else: 
        if str(model) == 'RandomForestRegressor()':
            hyperP = dict(n_estimators=[100, 300, 500, 800], 
                        max_depth=[None, 5, 8, 15, 25, 30],
                        min_samples_split=[2, 5, 10, 15, 100],
                        min_samples_leaf=[1, 2, 5, 10],
                        random_state = [seed])

        elif str(model) == 'GradientBoostingRegressor()':
            hyperP = dict(loss=['squared_error'], learning_rate=[0.1, 0.2, 0.3],
                        n_estimators=[100, 300, 500, 800], max_depth=[None, 5, 8, 15, 25, 30],
                        min_samples_split=[2],
                        min_samples_leaf=[1, 2],
                        random_state = [seed])

        gridF = GridSearchCV(model, hyperP, cv=3, verbose=1, n_jobs=-1)
        bestP = gridF.fit(X, y)

        params = bestP.best_params_
        print('Best hyperparameters:', params, '\n')

        return params






