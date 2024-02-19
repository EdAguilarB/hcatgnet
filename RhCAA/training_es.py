#torch
import torch
from torch.nn import MSELoss, BCEWithLogitsLoss
import torch_geometric
from torch.optim.lr_scheduler import ReduceLROnPlateau

#other packages
import os
import numpy as np
import argparse
import time
import random
from copy import deepcopy

#import custom functions from directory
import sys
sys.path.insert(0, 
                './Utils')

#custom packages
from Utils.model import GCN_loop
from Utils.dataset import ChiralLigands_regr, ChiralLigands_class
from Utils.utils import create_loaders, train, test, model_report, outer_summary_report, experiment_summary_report


PARSER = argparse.ArgumentParser(description="Perform nested cross validation for GNN for prediction of stereoselectivity of rhodium catalysed 1,4 Michael additions.")
PARSER.add_argument("-d", "--dir", dest="d", type=str, default="RhCAA/GNN",
                    help="Name of the directory where experiments will be saved.")
PARSER.add_argument("-e", "--exp", dest="e", type=str, default="r",
                    help="Experiment to perform. Accepted values are 'r' for regression and 'c' for classification")
PARSER.add_argument("-s", "--seed", type=int, dest="s", default=20232023,
                    help="Random seed for reproducibility.")
ARGS = PARSER.parse_args()


HYPERPARAMS = {}
# Process-related                  
HYPERPARAMS["batch_size"] = 40
HYPERPARAMS["epochs"] = 250
HYPERPARAMS["lr0"] = 0.01
HYPERPARAMS["betas"] = (0.9, 0.999)     
HYPERPARAMS["eps"] = 1e-9     
HYPERPARAMS["weight_decay"] = 0         
HYPERPARAMS["amsgrad"] = False


# scheduler
HYPERPARAMS['factor'] = .7
HYPERPARAMS['patience'] = 7
HYPERPARAMS['min_lr'] = 1e-08


# Model-related
HYPERPARAMS["dim"] = 64
HYPERPARAMS["improved"] = True 
HYPERPARAMS["n_conv"] = 2

HYPERPARAMS["splits"] = 10

if ARGS.e == 'r':
    HYPERPARAMS["loss_fn"] = MSELoss()
elif ARGS.e == 'c':
    HYPERPARAMS["loss_fn"] = BCEWithLogitsLoss()



def main():

    SEED = ARGS.s
    torch_geometric.seed_everything(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    np.random.RandomState(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    print('Initialising chiral ligands experiment using early stopping')

    current_dir = os.getcwd()
    save_dir = ARGS.d
    parent_dir = os.path.join(current_dir, save_dir)

    print("Saving the results in the parent directory: {}".format(parent_dir))

    os.makedirs(parent_dir, exist_ok=True)

    print('Model hyperparameters: \n', HYPERPARAMS, '\n')

    print('Performing experiment using the learning dataset as training, validation, and test set with inner and outer cross validation.')
    val_size = HYPERPARAMS["splits"]-1
    TOT_RUNS = HYPERPARAMS["splits"]*val_size

    print("Number of splits: {}".format(HYPERPARAMS["splits"]))
    print("Total number of runs: {}".format(TOT_RUNS))

    print("--------------------------------------------")

    # Select device (GPU/CPU)
    device_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("Device name: {} (GPU)".format(torch.cuda.get_device_name(0)))
        device_dict["name"] = torch.cuda.get_device_name(0)
        device_dict["CudaDNN_enabled"] = torch.backends.cudnn.enabled
        device_dict["CUDNN_version"] = torch.backends.cudnn.version()
        device_dict["CUDA_version"] = torch.version.cuda

    else:
        print("Device name: CPU")
        device_dict["name"] = "CPU"   

    if ARGS.e == 'r':
        print('Predicting quantitative selectivity of process.')
        monocyclic = ChiralLigands_regr(root = '{}/Data/learning/'.format(current_dir), filename= 'learning_folds.csv', include_fold=True)

    elif ARGS.e == 'c':
        print('Predicting side of addition of process.')
        monocyclic = ChiralLigands_class(root = '{}/Data/learning/'.format(current_dir), filename= 'learning_folds.csv', include_fold=True)

    print("Dataset type: ", type(monocyclic))
    print("Dataset length: ", monocyclic.len())
    print("Dataset node features: ", monocyclic.num_features)
    print("Dataset target: ", monocyclic.num_classes)
    print("Dataset sample: ", monocyclic[0])
    print("Sample  nodes: ", monocyclic[0].num_nodes)
    print("Sample  edges: ", monocyclic[0].num_edges, '\n')

    
    ncv_iterator = create_loaders(dataset = monocyclic, batch_size=HYPERPARAMS["batch_size"], folds=HYPERPARAMS["splits"])

    counter = 0
    metrics_all_train, metrics_all_val, metrics_all_test = [], [], []

    for outer in range(HYPERPARAMS["splits"]):

        os.makedirs("{}/{}".format(parent_dir, outer+1), exist_ok=True)

        loss_inner_test, loss_inner_train, loss_inner_val = [], [], []
        metrics_train, metrics_val, metrics_test = [], [], [] 

        for inner in range(val_size):

            val_best_loss = 1_000
            early_stop_counter = 0
            best_epoch = 0

            counter += 1

            train_loader, val_loader, test_loader = next(ncv_iterator)

            # Instantiate model, optimizer and lr-scheduler
            GCN_model = GCN_loop(monocyclic.num_features, 
                            embedding_size = HYPERPARAMS["dim"], 
                            gnn_layers = HYPERPARAMS["n_conv"], 
                            improved=HYPERPARAMS["improved"],
                            SEED=SEED,
                            task=ARGS.e).to(device)     
            
            optimiser = torch.optim.Adam(GCN_model.parameters(),
                                        lr=HYPERPARAMS["lr0"], 
                                        betas=HYPERPARAMS["betas"],
                                        eps=HYPERPARAMS["eps"], 
                                        weight_decay=HYPERPARAMS["weight_decay"], 
                                        amsgrad=HYPERPARAMS["amsgrad"])
            
            lr_scheduler = ReduceLROnPlateau(optimiser, 
                                             mode = 'min',
                                             factor=HYPERPARAMS["factor"],
                                             patience=HYPERPARAMS["patience"],
                                             min_lr=HYPERPARAMS["min_lr"])

            train_list, val_list, test_list = [], [], []       
            t0 = time.time()
            
            for epoch in range(HYPERPARAMS["epochs"]):

                if early_stop_counter <= 6:

                    torch.cuda.empty_cache()

                    #lr = lr_scheduler.optimizer.param_groups[0]['lr']        

                    train_loss, train_metrics = train(train_loader=train_loader, 
                                                 device=device,
                                                 loss_fn=HYPERPARAMS["loss_fn"], 
                                                 model=GCN_model, 
                                                 optimizer=optimiser,
                                                 task=ARGS.e) 
                    
                    val_loss, val_metrics = test(test_loader=val_loader,
                                                    device=device,
                                                    loss_fn=HYPERPARAMS["loss_fn"], 
                                                    model=GCN_model,
                                                    task=ARGS.e)
                    
                    #lr_scheduler.step(val_MAE)

                    test_loss, test_metrics = test(test_loader=test_loader,
                                                           device=device, 
                                                           loss_fn=HYPERPARAMS["loss_fn"], 
                                                           model=GCN_model,
                                                           task=ARGS.e,
                                                           test=True)
                    
                        
                    print('{}/{}-Epoch {:03d} | Train loss: {:.4f} % | Validation loss: {:.4f} % | '             
                        'Test loss: {:.4f} %'.format(counter, TOT_RUNS, epoch, train_loss, val_loss, test_loss))
                    

                    if epoch%5==0:

                        lr_scheduler.step(val_loss)
                        test_list.append(test_loss)
                        train_list.append(train_loss)
                        val_list.append(val_loss)
                    
                        if val_loss<= val_best_loss:

                            train_best_loss = train_loss
                            train_best_metrics = train_metrics

                            val_best_loss = val_loss
                            val_best_metrics = val_metrics

                            test_best_loss = test_loss
                            test_best_metrics = test_metrics

                            best_epoch = epoch

                            print('New best validation loss: {:.4f} at epoch {}'.format(val_best_loss, best_epoch))

                            early_stop_counter = 0

                            best_model_params = deepcopy(GCN_model.state_dict())

                        else:
                            early_stop_counter += 1

                    


                    if epoch == HYPERPARAMS["epochs"] -1:
                        print('Max quantity of epochs reached.')
                        loss_inner_test.append(test_best_loss)
                        loss_inner_train.append(train_best_loss)
                        loss_inner_val.append(val_best_loss)

                        metrics_train.append(train_best_metrics)
                        metrics_val.append(val_best_metrics)
                        metrics_test.append(test_best_metrics)


                else:

                    print('Early stopping due to no improvement.')
                    loss_inner_test.append(test_best_loss)
                    loss_inner_train.append(train_best_loss)
                    loss_inner_val.append(val_best_loss)

                    metrics_train.append(train_best_metrics)
                    metrics_val.append(val_best_metrics)
                    metrics_test.append(test_best_metrics)

                    break

            print("-----------------------------------------------------------------------------------------")

            training_time = (time.time() - t0)/60  
            print("Training time: {:.2f} min".format(training_time))
            device_dict["training_time"] = training_time

            print("-----------------------------------------------------------------------------------------\n")

            model_report(model_name="{}_{}_model".format(outer+1, inner+1),
                         outer = "{}".format(outer+1),
                         model_path="{}".format(parent_dir),
                         model=GCN_model,
                         model_params=best_model_params,
                         loaders=[train_loader, val_loader, test_loader],
                         loss_lists=[train_list, val_list, test_list],
                         best_epoch=best_epoch,
                         task = ARGS.e)



            del GCN_model, optimiser,  train_loader, val_loader, test_loader, best_model_params
            if device == "cuda":
                torch.cuda.empty_cache()

        

        metrics_train = np.array(metrics_train)
        metrics_val = np.array(metrics_val)
        metrics_test = np.array(metrics_test)

        outer_summary_report(run_name='outer_summary_{}'.format(outer+1),
                            run_path='{}/{}'.format(parent_dir, outer+1),
                            best_metrics=(metrics_train, metrics_val, metrics_test))
        
        metrics_all_train.append(metrics_train)
        metrics_all_val.append(metrics_val)
        metrics_all_test.append(metrics_test)
        

    experiment_summary_report(experiment_name='summary_all_experiments',
                              run_path='{}'.format(parent_dir),
                              metrics=(metrics_all_train,metrics_all_val, metrics_all_test),
                              hyperparams=HYPERPARAMS,
                              device=device_dict)
    
    return 'Experiment has finalised.'

    

if __name__ == "__main__":  
    main()  