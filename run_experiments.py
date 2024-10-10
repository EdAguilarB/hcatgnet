import os
from scripts_experiments.train_GNN import train_network_nested_cv
from scripts_experiments.train_TML import train_tml_model_nested_cv
from scripts_experiments.predict_test import predict_final_test
from scripts_experiments.compare_gnn_tml import plot_results
from scripts_experiments.explain_gnn import denoise_graphs, GNNExplainer_node_feats, shapley_analysis
from options.base_options import BaseOptions
import os

def run_all_exp():

    opt = BaseOptions().parse()

    if opt.train_GNN:
        if not os.path.exists(os.path.join(opt.log_dir_results, opt.filename[:-4], 'results_GNN')):
            train_network_nested_cv(opt)
        else:
            print('GNN model has already been trained')
    
    if opt.train_tml:
        if not os.path.exists(os.path.join(opt.log_dir_results, opt.filename[:-4], 'results_TML', opt.tml_algorithm, opt.descriptors)):
            train_tml_model_nested_cv(opt)
        else:
            print('TML model has already been trained')

    if opt.predict_unseen:
        if not os.path.exists(os.path.join(opt.log_dir_results, opt.filename_final_test[:-4], 'results_TML', opt.tml_algorithm, opt.descriptors)):
            predict_final_test(opt)
        else:
            print('Prediction of unseen data has already been done')

    if opt.compare_models:
        if not os.path.exists(os.path.join(opt.log_dir_results, opt.filename[:-4], f'GNN_vs_{opt.tml_algorithm}')):
            plot_results(os.path.join(opt.log_dir_results, opt.filename[:-4]), opt)
        else:
            print(f'GNN and TML ({opt.tml_algorithm}) Models have already been compared for {opt.filename[:-4]} dataset.')

        if not os.path.exists(os.path.join(opt.log_dir_results, opt.filename_final_test[:-4], f'GNN_vs_{opt.tml_algorithm}')):
            plot_results(exp_dir=os.path.join(opt.log_dir_results, opt.filename_final_test[:-4]))
        else:
            print(f'GNN and TML ({opt.tml_algorithm}) Models have already been compared for {opt.filename_final_test[:-4]} dataset.')

    if opt.denoise_graph:
        denoise_graphs(opt, exp_path=os.path.join(os.getcwd(), opt.log_dir_results, opt.filename[:-4], 'results_GNN'))

    if opt.GNNExplainer:
        GNNExplainer_node_feats(opt, exp_path=os.path.join(os.getcwd(), opt.log_dir_results, opt.filename[:-4], 'results_GNN'))

    if opt.shapley_analysis:
        shapley_analysis(opt, exp_path=os.path.join(os.getcwd(), opt.log_dir_results, opt.filename[:-4], 'results_GNN'))


if __name__ == '__main__':
    run_all_exp()

