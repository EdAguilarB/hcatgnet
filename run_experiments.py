import os
from scripts_experiments.train_GNN import train_network_nested_cv
from scripts_experiments.train_TML import train_tml_model_nested_cv
from scripts_experiments.predict_test import predict_final_test
from scripts_experiments.compare_gnn_tml import plot_results
from scripts_experiments.explain_gnn import explain_model
from options.base_options import BaseOptions

def run_all_exp():

    opt = BaseOptions().parse()

    if opt.train_GNN:
        train_network_nested_cv()
    
    if opt.train_tml:
        train_tml_model_nested_cv()

    if opt.predict_unseen:
        predict_final_test()

    if opt.compare_models:
        plot_results(exp_dir=os.path.join(os.getcwd(), opt.log_dir_results, 'learning_set'))
        plot_results(exp_dir=os.path.join(os.getcwd(), opt.log_dir_results, 'final_test'))

    if opt.explain_GNN:
        explain_model(exp_path=os.path.join(os.getcwd(), opt.log_dir_results, 'learning_set', 'results_GNN'))


if __name__ == '__main__':
    run_all_exp()

