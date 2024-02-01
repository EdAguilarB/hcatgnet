#model related
import torch

#data structures
import numpy as np
import pandas as pd
import csv

#file management
import os
import re
import joblib

#useful functions
from copy import copy, deepcopy
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, \
    accuracy_score, precision_score, recall_score, f1_score
from math import sqrt, ceil
from datetime import date, datetime

from icecream import ic

from scipy.stats import pearsonr

#ploting function
from ploting_functions import *

SEED = 123


def split_list(a: list, n: int):
    """
    Split a list into n chunks (for nested cross-validation)
    Args:
        a(list): list to split
        n(int): number of chunks
    Returns:
        (list): list of chunks
    """
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def create_loaders(dataset, batch_size, folds = 10, num_points=None):

    """
    Creates training, validation and testing loaders for cross validation and
    inner cross validation training-evaluation processes.
    Args:
    dataset: pytorch geometric datset
    batch_size (int): size of batches
    val (bool): whether or not to create a validation set
    folds (int): number of folds to be used
    num points (int): number of points to use for the training and evaluation process

    Returns:
    (tuple): DataLoaders for training, validation and test set

    """
    folds = [[] for _ in range(folds)]
    for data in dataset:
        folds[data.fold].append(data)

    if num_points:
        k, m = divmod(num_points, len(folds))
        for f, fold in enumerate(folds):
            folds[f] = fold[:k+1 if f < m else k]
        
    for outer in range(len(folds)):
        proxy = copy(folds)
        test_loader = DataLoader(proxy.pop(outer), batch_size=batch_size, shuffle=False)
        for inner in range(len(proxy)):  # length is reduced by 1 here
            proxy2 = copy(proxy)
            val_loader = DataLoader(proxy2.pop(inner), batch_size=batch_size, shuffle=False)
            flatten_training = [item for sublist in proxy2 for item in sublist]  # flatten list of lists
            train_loader = DataLoader(flatten_training, batch_size=batch_size, shuffle=True)
            yield deepcopy((train_loader, val_loader, test_loader))


def split_percentage(splits: int, test: bool=True):
    """Return split percentage of the train, validation and test sets.

    Args:
        split (int): number of initial splits of the entire initial dataset

    Returns:
        a, b, c: train, validation, test percentage of the sets.
    """
    if test:
        a = int(100 - 200 / splits)
        b = ceil(100 / splits)
        return a, b, b
    else:
        return int((1 - 1/splits) * 100), ceil(100 / splits)


def metrics_plot(df:pd.DataFrame):

    y_true_c = np.where(df['real_top']>50,1,0)
    y_predicted_c = np.where(df['predicted_top']>50,1,0)

    accuracy = accuracy_score(y_true=y_true_c, y_pred=y_predicted_c)
    precision = precision_score(y_true=y_true_c, y_pred=y_predicted_c)
    recall = recall_score(y_true=y_true_c, y_pred=y_predicted_c)


    y_true_r = df.loc[abs(df['Error'])<50, 'real_top']
    y_predicted_r = df.loc[abs(df['Error'])<50, 'predicted_top']


    mae = mean_absolute_error(y_true=y_true_r, y_pred=y_predicted_r)
    rmse = sqrt(mean_squared_error(y_true=y_true_r, y_pred=y_predicted_r))  
    r2 = pearsonr(y=y_true_r, x=y_predicted_r)[0]**2

    metrics_c = [accuracy, precision, recall, r2]

    metrics_e = [mae, rmse]

    return np.array(metrics_c), np.array(metrics_e)




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


def train(model,
          device,
          train_loader,
          optimizer,
          loss_fn,
          task):
    
    model.train()

    all_preds = []
    all_y = []
    total_loss = 0

    for batch in train_loader:

        # Use GPU
        batch.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Passing the node features and the connection info
        pred = model(batch.x.float(),  
                        batch.edge_index,
                        batch.batch)
        
        # Calculating the loss and gradients
        if task == 'r':
            loss = torch.sqrt(loss_fn(pred, torch.unsqueeze(batch.y.float(), dim = 1)))
            all_y.append(batch.y.cpu().detach().numpy())
            all_preds.append(pred.cpu().detach().numpy())

        elif task == 'c':
            loss = loss_fn(pred, torch.unsqueeze(batch.category.float(), dim = 1))
            all_y.append(batch.category.cpu().detach().numpy())
            all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))


        loss.backward()

        # Update using the gradients
        optimizer.step() 

        total_loss += loss.item() * batch.num_graphs

    
    total_loss /= len(train_loader.dataset)

    all_preds = np.concatenate(all_preds).ravel()
    all_y = np.concatenate (all_y).ravel() 


    metrics = calculate_metrics(all_y, all_preds, task=task)

    return total_loss, metrics



def test(model,
        device,
        test_loader,
        loss_fn,
        task,
        test=False):
    
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0

    for batch in test_loader:

        batch.to(device)
        pred = model(batch.x.float(),  
                        batch.edge_index, 
                        batch.batch)
        
        if task == 'r':
            loss = torch.sqrt(loss_fn(pred, torch.unsqueeze(batch.y.float(), dim = 1)))
            all_labels.append(batch.y.cpu().detach().numpy())
            all_preds.append(pred.cpu().detach().numpy())

        elif task == 'c':
            loss = loss_fn(pred, torch.unsqueeze(batch.category.float(), dim = 1))
            all_labels.append(batch.category.cpu().detach().numpy())
            all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))


        total_loss += loss.item() *batch.num_graphs
    

    total_loss /= len(test_loader.dataset)
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    
    metrics = calculate_metrics(all_labels, all_preds, all_metrics=test, task=task)

    return total_loss, metrics



def pred_sets(loader,
              model, 
              task):
    
    model.to('cpu')

    y_pred, y_true, idx = [], [], []
    
    for batch in loader:
        with torch.no_grad():
            batch = batch.to('cpu')
            pred = model(batch.x.float(),  
                        batch.edge_index, 
                        batch.batch)

            y_pred.append(pred.cpu().detach().numpy())
            y_true.append(batch.y.cpu().detach().numpy())
            idx.append(batch.idx.cpu().detach().numpy())

    y_pred = np.concatenate(y_pred).ravel()
    y_true = np.concatenate(y_true).ravel()

    if task == 'c':
        y_pred = np.where(y_pred<0, 0, 1)
        y_true = np.where(y_true<50, 0, 1)

    idx = np.concatenate(idx).ravel()

    return y_pred, y_true, idx


def model_report(model_name,
                 outer, 
                 model_path,
                 model,
                 model_params,
                 loaders,
                 loss_lists,
                 best_epoch, 
                 task = 'r',
                 save_all = True):
    
    # 1) Get time of the run
    today = date.today()
    today_str = str(today.strftime("%d-%b-%Y"))
    time = str(datetime.now())[11:]
    time = time[:8]
    run_period = "{}, {}\n".format(today_str, time)

    # 2) Unfold  train/val/test dataloaders
    train_loader, val_loader, test_loader = loaders[0], loaders[1], loaders[2]
    N_train, N_val, N_test = len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)
    N_tot = N_train + N_val + N_test 
    
    log_dir = "{}/{}/{}".format(model_path, outer, model_name)
    
    #this creates a directory to store results if the directory does not exist
    try:
        os.makedirs("{}".format(log_dir))
    except FileExistsError:
        model_name = input("The name defined already exists in the provided directory: Provide a new one: ")
        log_dir = "{}/{}/{}".format(model_path, outer, model_name)
        os.mkdir("{}".format(log_dir))

    # loss trend during training
    train_list = loss_lists[0]
    val_list = loss_lists[1] 
    test_list = loss_lists[2]


    # 9) Save dataloaders for future use
    if save_all == True:
        torch.save(train_loader, "{}/train_loader.pth".format(log_dir))
        torch.save(val_loader, "{}/val_loader.pth".format(log_dir))
    torch.save(test_loader, "{}/test_loader.pth".format(log_dir))

    # 10) Save model architecture and parameters

    if task == 'r':
        if save_all == True:
            torch.save(model, "{}/model_r.pth".format(log_dir))             # Save model architecture
            torch.save(model_params, "{}/GNN_params_r.pth".format(log_dir))  # Save model parameters
        loss_function = 'RMSE_%'
        training_file = 'training_regression'
        metrics = ['R2', 'MAE', 'RMSE', 'Mean Bias Error', \
                   'Mean absolute percentage error', 'Error Standard Deviation']

    elif task == 'c':
        if save_all == True:
            torch.save(model, "{}/model_c.pth".format(log_dir))             # Save model architecture
            torch.save(model_params, "{}/GNN_params_c.pth".format(log_dir))  # Save model parameters
        loss_function = 'BCE'
        training_file = 'training_classification'
        metrics = ['Accuracy', 'Precision', 'Recall', 'f1']

    

    # 12) Store train_list, val_list, and lr_list as .csv file
    if train_list is not None and val_list is not None and test_list is not None:
        with open('{}/{}.csv'.format(log_dir, training_file), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Epoch", "Train_{}".format(loss_function), "Val_{}".format(loss_function), "Test_{}".format(loss_function)])
            for i in range(len(train_list)):
                writer.writerow([(i+1)*5, train_list[i], val_list[i], test_list[i]])

        create_training_plot(df='{}/{}.csv'.format(log_dir, training_file), save_path='{}'.format(log_dir))

    
    file1 = open("{}/performance.txt".format(log_dir), "w")
    file1.write(run_period)
    file1.write("---------------------------------------------------------\n")
    file1.write("GNN TRAINING AND PERFORMANCE\n")
    file1.write("Best epoch: {}\n".format(best_epoch))
    file1.write("Dataset Size = {}\n".format(N_tot))
    file1.write("***************\n")

    model.load_state_dict(model_params)

    y_pred, y_true, idx = pred_sets(loader=train_loader, model=model, task=task)
    metric_vals = calculate_metrics(y_true=y_true, y_predicted=y_pred, all_metrics=True, task=task)

    # Performance Report
    file1.write("Training set\n")
    file1.write("Training set size = {}\n".format(N_train))

    for name, value in zip(metrics, metric_vals):
        file1.write("{} Train = {:.3f} \n".format(name, value))


    file1.write("***************\n")

    y_pred, y_true, idx = pred_sets(loader=val_loader, model=model, task=task)
    metric_vals = calculate_metrics(y_true=y_true, y_predicted=y_pred, all_metrics=True, task=task)

    file1.write("Validation set\n")
    file1.write("Validation set size = {}\n".format(N_val))

    for name, value in zip(metrics, metric_vals):
        file1.write("{} Validation = {:.3f} \n".format(name, value))


    file1.write("***************\n")

    y_pred, y_true, idx = pred_sets(loader=test_loader, model=model, task=task)

    side_pred = np.where(y_pred>50,1,0)
    side_true = np.where(y_true>50,1,0)

    metrics_class = calculate_metrics(y_true=side_true, y_predicted=side_pred, task='c')


    file1.write("Test set\n")
    file1.write("Test set size = {}\n".format(N_test))

    metrics_class_names = ['Accuracy', 'Precision',  'Recall', 'F1']

    for name, value in zip(metrics_class_names, metrics_class):
        file1.write("{} Test = {:.3f} \n".format(name, value))


    error = abs(y_true-y_pred)
    y_true = y_true[error<50]
    y_pred = y_pred[error<50]
    idx = idx[error<50]

    metric_vals = calculate_metrics(y_true=y_true, y_predicted=y_pred, all_metrics=True, task=task)


    file1.write("Test Set Total Correct Side  Predictions = {}\n".format(len(y_true)))


    for name, value in zip(metrics, metric_vals):
        file1.write("{} Test = {:.3f} \n".format(name, value))


    file1.write("---------------------------------------------------------\n")
    


    create_st_parity_plot(real = y_true, predicted = y_pred, figure_name = model_name, save_path = "{}".format(log_dir))
    create_it_parity_plot(y_true, y_pred, idx, '{}.html'.format(model_name), "{}".format(log_dir))
    


    if task == 'r':
        file1.write("OUTLIERS (TEST SET)\n")

        error_test = [(y_pred[i] - y_true[i]) for i in range(len(y_pred))] 
        abs_error_test = [abs(error_test[i]) for i in range(len(y_pred))]
        std_error_test = np.std(error_test)

        outliers_list, outliers_error_list, index_list = [], [], []
        counter = 0

        for sample in range(len(y_pred)):
            if abs_error_test[sample] >= 3 * std_error_test:  
                counter += 1
                outliers_list.append(idx[sample])
                outliers_error_list.append(error_test[sample])
                index_list.append(sample)
                if counter < 10:
                    file1.write("0{}) {}    Error: {:.4f} %    (index={})\n".format(counter, idx[sample], error_test[sample], sample))
                else:
                    file1.write("{}) {}    Error: {:.4f} %s    (index={})\n".format(counter, idx[sample], error_test[sample], sample))

    file1.close()

    return "Model saved in {}".format(log_dir)


def outer_summary_report(run_name,
                          run_path,
                          best_metrics):
        
    set_names = ['Training set', 'Validation set', 'Test set']

    file1 = open("{}/{}.txt".format(run_path, run_name), "w")
        
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
                              hyperparams, 
                              device):
    
        # 7) Store info of device on which model training has been performed
    if device != None:
        with open('{}/device.txt'.format(run_path), 'w') as f:
            print(device, file=f)
    
        # 11) Store Hyperparameters dict from input file
    with open('{}/hyper_params.txt'.format(run_path), 'w') as g:
        print(hyperparams, file=g)
    
    set_names = ['Training set', 'Validation set', 'Test set']

    file1 = open("{}/{}.txt".format(run_path, experiment_name), "w")

    if device is not None:
        file1.write("Device = {}\n".format(device["name"]))
        file1.write("Training time = {:.2f} min\n".format(device["training_time"]))

    file1.write("---------------------------------------------------------\n")
    file1.write("GNN ARCHITECTURE\n")
    file1.write("Number of convolutional layers = {}\n".format(hyperparams["n_conv"]))
    file1.write("Size of hidden feature vector = {}\n".format(hyperparams["dim"]))
    file1.write("GCN improved = {}\n".format(hyperparams["improved"]))
    file1.write("---------------------------------------------------------\n")
    file1.write("TRAINING PROCESS\n")
    file1.write("Data Split (Train/Val/Test) = {}-{}-{} %\n".format(*split_percentage(hyperparams["splits"])))
    file1.write("Batch size = {}\n".format(hyperparams["batch_size"]))
    file1.write("Optimizer = Adam\n")                                            
    file1.write("Initial learning rate = {}\n".format(hyperparams["lr0"]))
    file1.write("Loss function = {}\n".format(hyperparams["loss_fn"]))
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
    

def predict(model_arch, model_params, data_loader):

    device = "cpu"

    model = torch.load(model_arch, map_location=torch.device(device))
    model.load_state_dict(torch.load(model_params, map_location=torch.device(device)))

    with torch.no_grad():

        predictions = []
        y = []
        indexes = []

        for batch in data_loader:

            batch.to(device)

            out = model(batch.x,
                        batch.edge_index,
                        batch.batch)
            
            predictions.append(out.cpu().detach().numpy())
            y.append(batch.y.cpu().detach().numpy())
            indexes.append(batch.idx.cpu().detach().numpy())

        predictions = np.concatenate(predictions).ravel()
        y = np.concatenate(y).ravel()
        indexes = np.concatenate(indexes).ravel()


    df = pd.DataFrame({'real_top': y, 'predicted_top': predictions, 'Index': indexes})

    return df


def test_summary_outer(metrics,
                       save_dir,
                       outer):


    metrics_mean = np.mean(metrics, axis = 0)
    metrics_std = np.std(metrics, axis = 0)

    file1 = open("{}/outer_summary_{}.txt".format(save_dir, outer), "w")

    for i, row in enumerate(metrics):
        file1.write("*********************\n")
        file1.write("Metrics using fold {} as validation set\n".format(i+1))
        file1.write("R2 score: {:.3f}\n".format(row[0]))
        file1.write("Mean absolute error: {:.3f}\n".format(row[1]))
        file1.write("Root mean squared error: {:.3f}\n".format(row[2]))
        file1.write("Mean absolute percentage error: {:.3f}\n".format(row[3]))
        file1.write("Mean bias error: {:.3f}\n".format(row[4]))
        file1.write("Error std: {:.3f}\n".format(row[5]))

    file1.write("*********************\n")
    file1.write('Test set\n')
    file1.write('Summary of results reported as mean ± std:\n')
    file1.write("R2 = {:.3f} ± {:.3f}\n".format(metrics_mean[0], metrics_std[0]))
    file1.write("MAE = {:.3f} ± {:.3f}\n".format(metrics_mean[1], metrics_std[1]))
    file1.write("RMSE = {:.3f} ± {:.3f}\n".format(metrics_mean[2], metrics_std[2]))
    file1.write("Mean absolute percentage error: {:.3f} ± {:.3f}\n".format(metrics_mean[3], metrics_std[3]))
    file1.write("Mean bias error: {:.3f} ± {:.3f}\n".format(metrics_mean[4], metrics_std[4]))
    file1.write("Error std: {:.3f} ± {:.3f}\n".format(metrics_mean[5], metrics_std[5]))

    file1.close()

    return 'Performance file was saved in {}'.format(save_dir)




def extract_metrics(directory, metric, file = "summary_all_experiments.txt", content ='Test set'):

    
    # List all files in the given directory
    file = os.path.join(directory, file) 
    
    # Define regular expressions to match metric lines
    mae_pattern = re.compile(r"MAE = (\d+\.\d+) ± (\d+\.\d+)")
    rmse_pattern = re.compile(r"RMSE = (\d+\.\d+) ± (\d+\.\d+)")
    r2_pattern = re.compile(r"R2 = (\d+\.\d+) ± (\d+\.\d+)")
    
    with open(os.path.join(file), 'r') as f:
        content = f.read()
            
            # Split the content by '*' to separate different sets
    sets = content.split('*')


    for set_content in sets:
                # Check if "Test set" is in the set content
        if 'Test set' in set_content:

            # Extract MAE and RMSE values using regular expressions
            if metric == 'MAE':
                mae_match = mae_pattern.search(set_content)
                mae_mean = float(mae_match.group(1))
                mae_std = float(mae_match.group(2))
                return [mae_mean, mae_std]

            elif metric == 'RMSE':
                rmse_match = rmse_pattern.search(set_content)
                rmse_mean = float(rmse_match.group(1))
                rmse_std = float(rmse_match.group(2))
                return [rmse_mean, rmse_std]

            elif metric == 'R2':
                r2_match = r2_pattern.search(set_content)
                r2_mean = float(r2_match.group(1))
                r2_std = float(r2_match.group(2))
                return [r2_mean, r2_std]
                    



def extract_metrics_all(directory, metric, file = "performance.txt"):

    
    # List all files in the given directory
    file = os.path.join(directory, file) 
    
    # Define regular expressions to match metric lines
    mae_pattern = re.compile(r"MAE Test = (\d+\.\d+)")
    rmse_pattern = re.compile(r"RMSE Test = (\d+\.\d+)")
    r2_pattern = re.compile(r"R2 Test = (\d+\.\d+)")
    
    with open(os.path.join(file), 'r') as f:
        content = f.read()
            
            # Split the content by '*' to separate different sets
    sets = content.split('*')


    for set_content in sets:
                # Check if "Test set" is in the set content
        if "Test set" in set_content:
                    # Extract MAE and RMSE values using regular expressions
            if metric == 'MAE':
                mae_match = mae_pattern.search(set_content)
                mae_val = float(mae_match.group(1))
                return mae_val

            elif metric == 'RMSE':
                rmse_match = rmse_pattern.search(set_content)
                rmse_val = float(rmse_match.group(1))
                return rmse_val

            elif metric == 'R2':
                r2_match = r2_pattern.search(set_content)
                r2_val = float(r2_match.group(1))
                return r2_val



def predict_gnn(model_arch, model_params, data_loader):

    device = "cpu"

    model = torch.load(model_arch, map_location=torch.device(device))
    model.load_state_dict(torch.load(model_params, map_location=torch.device(device)))

    with torch.no_grad():

        predictions = []
        y = []
        indexes = []

        for batch in data_loader:

            batch.to(device)

            out = model(batch.x,
                        batch.edge_index,
                        #batch.edge_attr,
                        batch.batch)
            
            predictions.append(out.cpu().detach().numpy())
            y.append(batch.y.cpu().detach().numpy())
            indexes.append(batch.idx)

        predictions = np.concatenate(predictions).ravel()
        y = np.concatenate(y).ravel()
        indexes = np.concatenate(indexes).ravel()


    df = pd.DataFrame({'predicted_top': predictions, 'real_top': y, 'Index': indexes})

    return df


def predict_ap(model, data):


    model = joblib.load(model)

    data = pd.read_csv(data)

    descriptors = ['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6',
               'ER7', 'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4']

    predictions = model.predict(data[descriptors])

    y = data['%top']
    indexes = data['index']

    df = pd.DataFrame({'predicted_top': predictions, 'real_top': y, 'Index': indexes})

    return df



