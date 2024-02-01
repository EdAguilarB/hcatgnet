from utils import predict, create_st_parity_plot, create_it_parity_plot
import matplotlib.pyplot as plt
import os
import argparse

PARSER = argparse.ArgumentParser(description="Creates the parity plots of a series of points.")
PARSER.add_argument("-i", "--input", type=str, dest="i", 
                        help="Input toml file with hyperparameters for the nested cross validation.")
PARSER.add_argument("-d", "--dir", dest="d", type=str, default="Chiral_ligands_exp",
                    help="Name of the directory where experiments will be saved.")

ARGS = PARSER.parse_args()

def main():

    current_dir = os.getcwd()
    save_dir = ARGS.d
    parent_dir = os.path.join(current_dir, save_dir)

    plot_dir = os.path.join(parent_dir, 'plots')
    os.mkdir(plot_dir)

    

    results_dir = [f'{parent_dir}/{i}' for i in range(1, 11)]

    models = [f'{i}_1_model' for i in range(1,11)]

    model_dir = []
    params_dir = []
    loader_dir = []

    for result, model in zip(results_dir, models):
        model_dir.append(os.path.join(result,model,'model.pth'))
        params_dir.append(os.path.join(result,model,'GNN_params.pth'))
        loader_dir.append(os.path.join(result,model,'test_loader.pth'))
    

    for i in range(10):

        parity = predict(model_arch=model_dir[i], model_params=params_dir[i], loader_path=loader_dir[i])

        create_st_parity_plot(parity.iloc[:,0], parity.iloc[:,1], f'parity_plot_fold_{i}', '/home/pcxea2/experiments/ChiralLigands/parity')
        create_it_parity_plot(parity.iloc[:,0], parity.iloc[:,1], parity.iloc[:,2], f'parity_plot_fold_{i}.html', '/home/pcxea2/experiments/ChiralLigands/parity')

    print('All plots have been created and saved.')


if __name__ == "__main__":  
    main()  


