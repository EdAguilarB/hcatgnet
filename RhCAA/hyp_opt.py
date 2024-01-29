import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import SAGEConv, GATv2Conv, GraphMultisetTransformer
from torch.nn import ReLU, Tanh, Sigmoid  
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray
import pandas as pd
from model import *
from dataset_copy import *
from sklearn.model_selection import StratifiedKFold
from math import sqrt

torch_geometric.seed_everything(23)

HYPERPARAMS = {}
# Process-related                  
HYPERPARAMS["batch_size"] = tune.choice([16,32,64])           
HYPERPARAMS["epochs"] = 1500               
HYPERPARAMS["loss_function"] = torch.nn.MSELoss()  
HYPERPARAMS["lr0"] = tune.grid_search([0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])       
HYPERPARAMS["patience"] = tune.grid_search([1,3,5,7])        
HYPERPARAMS["factor"] = tune.grid_search([0.5, 0.7, 0.9])
HYPERPARAMS["minlr"] = 1e-8             
HYPERPARAMS["betas"] = (0.9, 0.999)     
HYPERPARAMS["eps"] = tune.grid_search([1e-8, 1e-9])               
HYPERPARAMS["weight_decay"] = 0         
HYPERPARAMS["amsgrad"] = tune.grid_search([True, False])

# Model-related
HYPERPARAMS["dim"] = tune.grid_search([16, 32, 64, 128, 256])     
HYPERPARAMS["improved"] = tune.choice([True, False])  
HYPERPARAMS["n_conv"] = tune.grid_search([1, 2, 3, 4, 5])           



def get_hyp_space(config: dict):
    """Return the total number of possible hyperparameters combinations.
    Args:
        config (dict): Hyperparameter configuration setting
    """
    x = 1
    counter = 0
    for key in list(config.keys()):
        if type(config[key]) == tune.search.sample.Categorical:
            x *= len(config[key])
            counter += 1

    #####
    #counter returns the number of tunable parameters and 
    #x returns the total possible combinations of hyperparameters
    #####

    return counter, x


def train_loop(model,
               device:str,
               train_loader: DataLoader,
               optimizer,
               loss_fn):

    model.train()  

    rmse_all, mae_all = 0, 0

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()                     # Set gradients of all tensors to zero
    
        pred = model(batch.x.float(), batch.edge_index, batch.batch) #predict on training data

        loss = torch.sqrt(loss_fn(pred, torch.unsqueeze(batch.y, dim=1))) #calculates loss of batch 

        mae = F.l1_loss(pred, torch.unsqueeze(batch.y, dim=1))    # For comparison with val/test data

        loss.backward()                           # Get gradient of loss function wrt parameters

        rmse_all += (loss.item()**2) * batch.num_graphs
        mae_all += mae.item() * batch.num_graphs

        optimizer.step()                          # Update model parameters

    rmse_all /= len(train_loader.dataset)
    rmse_all = sqrt(rmse_all)
    mae_all /= len(train_loader.dataset)
    
    return mae_all, rmse_all

def test_loop(model,
              loader: DataLoader,
              device: str,
              loss_fn,
              std: float=None,
              mean: float=None, 
              scaled_graph_label: bool= False, 
              verbose: int=0) -> float:

    model.eval()   
    mae_all = 0
    mse_all = 0

    for batch in loader:
        batch = batch.to(device) 
        pred = model(batch.x.float(), batch.edge_index, batch.batch)

        mae = F.l1_loss(pred, torch.unsqueeze(batch.y, dim=1))
        loss = loss_fn(pred, torch.unsqueeze(batch.y, dim=1))

        mae_all += mae.item() * batch.num_graphs
        mse_all += loss.item() * batch.num_graphs

        
    mae_all /= len(loader.dataset)   
    mse_all /= len(loader.dataset)
    rmse_all = sqrt(mse_all)  

    if verbose == 1:
        print("Dataset size = {}".format(len(loader.dataset)))
        print("Mean Absolute Error = {} %".format(mae_all))
        print("Root Mean Squared Error = {} %".format(rmse_all))
    return mae_all, rmse_all


def train_function(config: dict):

    ligands = ChiralLigands(root = '/home/pcxea2/experiments/ChiralLigands/Ligands_augmentation', filename = 'data_aug.csv')

    categories = np.array([ligands[i].category.detach().numpy() for i in range(len(ligands))])
    x = [ligands[i].x.detach().numpy() for i in range(len(ligands))]

    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=23)
    train_idx = []
    test_idx = []

    for i, (train, test) in enumerate(folds.split(x, categories)):
        train_idx.append(train)
        test_idx.append(test)

    train_loader = DataLoader(ligands[train_idx[0]], batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(ligands[test_idx[0]], batch_size=config['batch_size'], shuffle=False)



    # Define device, model, optimizer and lr-scheduler  
    device = "cuda" if torch.cuda.is_available() else "cpu"        

    #model contains the architecture of the model given the hyperparameters
    model = GCN_loop(ligands.num_features, 
                    embedding_size = config["dim"], 
                    gnn_layers=config["n_conv"], 
                    improved=config["improved"]).to(device)
    print(model)


    #Adam used as optimiser   
    optimizer = Adam(model.parameters(),
                     lr=config["lr0"], 
                     betas=config["betas"],
                     eps=config["eps"], 
                     weight_decay=config["weight_decay"], 
                     amsgrad=config["amsgrad"])
    
    #scheduler used in case of not improvement during training
    lr_scheduler = ReduceLROnPlateau(optimizer,
                                     mode='min',
                                     factor=config["factor"],
                                     patience=config["patience"],
                                     min_lr=config["minlr"])
    # Training process    
    train_list, val_list = [], []

    for iteration in range(1, config["epochs"]+1):
        torch.cuda.empty_cache()
        lr = lr_scheduler.optimizer.param_groups[0]['lr']

        train_MAE, train_rmse = train_loop(model, device, train_loader, optimizer, config["loss_function"])  

        val_MAE, val_rmse = test_loop(model, val_loader, device,loss_fn=config["loss_function"], verbose=0)
        lr_scheduler.step(val_MAE)  

        if iteration %5 ==0:
            print('Epoch {:03d}  Train MAE: {:.6f} %  Train RMSE: {:.6f} % '
                .format(iteration, train_MAE, train_rmse))  
            print('Val MAE: {:.6f} %  Val RMSE: {:.6f} % '
                .format(val_MAE, val_rmse))
                           
        train_list.append(train_MAE)
        val_list.append(val_MAE)

        tune.report(Val_loss=val_rmse, epoch=iteration)
    


hypopt_scheduler = ASHAScheduler(time_attr="epoch",
                                 metric="Val_loss",
                                 mode="min",
                                 grace_period=15,
                                 reduction_factor=4,
                                 max_t=1000,
                                 brackets=1)


def main():

    ray.init(ignore_reinit_error=True) 

    result = tune.run(train_function,
                      name='--output',
                      time_budget_s=60,
                      config=HYPERPARAMS,
                      scheduler=hypopt_scheduler,
                      resources_per_trial={"cpu":0, "gpu":3},
                      num_samples=5, 
                      verbose=1,
                      log_to_file=True, 
                      storage_path=".",
                      raise_on_failed_trial=False)

    ray.shutdown()  
    best_config = result.get_best_config(metric="Val_loss", mode="min", scope="last")
    best_config_df = pd.DataFrame.from_dict(best_config, orient="index")    
    best_config_df.to_csv("./best_config_1.csv", sep=",")
    print(best_config)
    exp_df = result.results_df
    exp_df.to_csv("./summary.csv", sep=",") 


if __name__ == "__main__":  
    main()   