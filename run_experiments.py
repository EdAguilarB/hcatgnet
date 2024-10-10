import os
from scripts_experiments.train_GNN import train_network_nested_cv
from scripts_experiments.train_TML import train_tml_model_nested_cv
from scripts_experiments.predict_test import predict_final_test
from scripts_experiments.compare_gnn_tml import plot_results
from scripts_experiments.explain_gnn import denoise_graphs, GNNExplainer_node_feats, shapley_analysis
from options.base_options import BaseOptions

def run_all_exp():

    opt = BaseOptions().parse()

    if opt.train_GNN:
        train_network_nested_cv(opt)
    
    if opt.train_tml:
        train_tml_model_nested_cv(opt)

    if opt.predict_unseen:
        #predict_final_test()
        pass

    if opt.compare_models:
        plot_results(exp_dir=os.path.join(os.getcwd(), opt.log_dir_results, opt.filename[:-4]))
        #plot_results(exp_dir=os.path.join(os.getcwd(), opt.log_dir_results, 'final_test'))

    if opt.denoise_graph:
        denoise_graphs(opt, exp_path=os.path.join(os.getcwd(), opt.log_dir_results, opt.filename[:-4], 'results_GNN'))

    if opt.GNNExplainer:
        GNNExplainer_node_feats(opt, exp_path=os.path.join(os.getcwd(), opt.log_dir_results, opt.filename[:-4], 'results_GNN'))

    if opt.shapley_analysis:
        shapley_analysis(opt, exp_path=os.path.join(os.getcwd(), opt.log_dir_results, opt.filename[:-4], 'results_GNN'))


if __name__ == '__main__':
    run_all_exp()

