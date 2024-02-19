import argparse
import torch
import os
from os.path import join
import sys

sys.path.insert(0, 
                './Utils')


from Utils.utils import extract_metrics_all, predict_gnn, predict_ap, metrics_plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import violinplot, stripplot




PARSER = argparse.ArgumentParser(description="Plot performance of both models (atomistic potential and GNN)\
                                  as mean and standard deviation for each testing-fold.")

PARSER.add_argument("-g", "--gnn", dest="g", type=str, default="RhCAA/GNN",
                    help="Name of the directory where the GNN experiments are saved.")

PARSER.add_argument("-t", "--tm", dest="t", type=str, default="RhCAA/Traditional_ML/GradientBoosting",
                    help="Name of the directory where the traditional ML experiments are saved.")

PARSER.add_argument("-l", "--log", type=str, dest="l", default='RhCAA/Results',
                    help="Directory where the plots will be saved.")


ARGS = PARSER.parse_args()





def main():


    means_gnn_e = []
    stds_gnn_e = []

    means_gnn_c = []
    stds_gnn_c = []

    means_tml_e = []
    stds_tml_e = []

    means_tml_c = []
    stds_tml_c = []

    current_dir = os.getcwd()
    log_dir = os.path.join(current_dir, ARGS.l)

    print('Results will be saved in {} directory'.format(log_dir))

    os.makedirs(log_dir, exist_ok=True)

    results_all = pd.DataFrame(columns = ['Error', 'Index', 'Fold', 'Method', 'real_top', 'predicted_top'])

    for outer in range(1,11):

        c_gnn_outer = []
        e_gnn_outer = []

        c_tml_outer = []
        e_tml_outer = []


        gnn_current_outer = join(ARGS.g, f'{outer}')
        ap_current_outer = os.path.join(ARGS.t, f'{outer}')

        for inner in range(1,10):

            gnn_current_inner = join(gnn_current_outer, f'{outer}_{inner}_model')
            ap_current_inner = join(ap_current_outer, f'{outer}_{inner}_model')

            loader = torch.load(join(gnn_current_inner, 'test_loader.pth'))

            df_gnn = predict_gnn(model_arch=join(gnn_current_inner, 'model_r.pth'),\
                                 model_params=join(gnn_current_inner, 'GNN_params_r.pth'),\
                                  data_loader=loader)
            
            df_gnn['Error'] = df_gnn['real_top'] - df_gnn['predicted_top']
            df_gnn['Fold'] = outer
            df_gnn['Inner_Fold'] = inner
            df_gnn['Method'] = 'GNN'
            df_gnn = df_gnn[['Error', 'Index', 'Fold', 'Inner_Fold', 'Method', 'real_top', 'predicted_top']]

            gnn_c, gnn_e = metrics_plot(df_gnn)

            c_gnn_outer.append(gnn_c)            
            e_gnn_outer.append(gnn_e)
            

            df_ap = predict_ap(model = join(ap_current_inner,'model.sav'), \
                               data=join(ap_current_inner, 'test.csv'))

            df_ap['Error'] = df_ap['real_top'] - df_ap['predicted_top']
            df_ap['Fold'] = outer
            df_ap['Method'] = 'Atomistic Potential'
            df_ap['Inner_Fold'] = inner
            df_ap = df_ap[['Error', 'Index', 'Fold', 'Inner_Fold', 'Method', 'real_top', 'predicted_top']]


            tml_c, tml_e = metrics_plot(df_ap)            
            c_tml_outer.append(tml_c)
            e_tml_outer.append(tml_e)


            results_all = pd.concat([results_all, df_gnn, df_ap], axis = 0)


        means_gnn_e.append(np.mean(np.array(e_gnn_outer), axis=0))
        stds_gnn_e.append(np.std(np.array(e_gnn_outer), axis = 0))

        means_gnn_c.append(np.mean(np.array(c_gnn_outer), axis=0))
        stds_gnn_c.append(np.std(np.array(c_gnn_outer), axis = 0))

        means_tml_e.append(np.mean(np.array(e_tml_outer), axis = 0))
        stds_tml_e.append(np.std(np.array(e_tml_outer), axis = 0))

        means_tml_c.append(np.mean(np.array(c_tml_outer), axis = 0))
        stds_tml_c.append(np.std(np.array(c_tml_outer), axis = 0))

    
    means_gnn_regr = np.array(means_gnn_e)
    stds_gnn_regr = np.array(stds_gnn_e)

    means_gnn_class = np.array(means_gnn_c)
    stds_gnn_class = np.array(stds_gnn_c)

    means_tml_regr = np.array(means_tml_e)
    stds_tml_regr = np.array(stds_tml_e)

    means_tml_class = np.array(means_tml_c)
    stds_tml_class = np.array(stds_tml_c)


    folds = list(range(1, 11))
    index = np.arange(10)

    error_metrics = ['MAE', 'RMSE']

    
    minimum = min(np.min(np.array(means_gnn_regr) - np.array(stds_gnn_regr)), np.min(np.array(means_tml_regr) - np.array(stds_tml_regr)))
    maximum = max(np.max(np.array(means_gnn_regr) + np.array(stds_gnn_regr)), np.max(np.array(means_tml_regr) + np.array(stds_tml_regr))) 
    
    for i, metric in enumerate(error_metrics):
            
        bar_width = 0.35

        plt.bar(index, means_gnn_regr[:,i], bar_width, label='GNN Approach', yerr=stds_gnn_regr[:,i], capsize=5)
        plt.bar(index + bar_width, means_tml_regr[:,i], bar_width, label='Traditional ML Approach', yerr=stds_tml_regr[:,i], capsize=5)

        plt.ylim(minimum*.99, maximum *1.01)
        plt.xlabel('Fold Used as Test Set', fontsize = 16)

        label = f'Mean {metric} Value / %'
        plt.ylabel(label, fontsize = 16)

        plt.xticks(index + bar_width / 2, folds)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.savefig(os.path.join(log_dir, f'{metric}_GNN_vs_TML'), dpi=300, bbox_inches='tight')

        print('Plot {}_GNN_vs_TML has been saved in the directory {}'.format(metric,log_dir))

        plt.clf()


    c_metrics = ['Accuracy', 'Precision', 'Recall', 'R2']
    minimum = min(np.min(np.array(means_gnn_class) - np.array(stds_gnn_class)), np.min(np.array(means_tml_class) - np.array(stds_tml_class)))
    maximun = max(np.max(np.array(means_gnn_class) + np.array(stds_gnn_class)), np.max(np.array(means_tml_class) + np.array(stds_tml_class)), 1)

    for i, metric in enumerate(c_metrics):
            
        bar_width = 0.35

        plt.bar(index, means_gnn_class[:,i], bar_width, label='GNN Approach', yerr=stds_gnn_class[:,i], capsize=5)
        plt.bar(index + bar_width, means_tml_class[:,i], bar_width, label='Traditional ML Approach', yerr=stds_tml_class[:,i], capsize=5)

        plt.ylim(minimum*.99, maximun *1.01)

        plt.xlabel('Fold Used as Test Set', fontsize = 16)

        label = 'Mean $R^2$ Value' if metric == 'R2' else f'Mean {metric} Value'

        plt.ylabel(label, fontsize = 16)
        plt.xticks(index + bar_width / 2, folds)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.savefig(os.path.join(log_dir, f'{metric}_GNN_vs_TML'), dpi=300, bbox_inches='tight')

        print('Plot {}_GNN_vs_TML has been saved in the directory {}'.format(metric,log_dir))

        plt.clf()



    '''
    Error distribution plots and csv with predictions containing all predictions
    '''
    violinplot(data = results_all, x='Fold', y='Error', hue='Method', split=True, gap=.1, inner="quart", fill=False)

    plt.xlabel('Fold Used as Test Set', fontsize=18)
    plt.ylabel('$\%top_{real}-\%top_{predicted}$', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax= plt.gca()
    ax.get_legend().remove()

    plt.savefig(os.path.join(log_dir, f'Error_distribution_GNN_vs_TML_violin_plot'), dpi=300, bbox_inches='tight')
    plt.close()

    stripplot(data=results_all, x="Fold", y="Error", hue="Method", size=3, dodge=True, jitter=True, marker='D', alpha=.3)

    plt.xlabel('Fold Used as Test Set', fontsize=18)
    plt.ylabel('$\%top_{real}-\%top_{predicted}$', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax= plt.gca()
    ax.get_legend().remove()
    
    plt.savefig(os.path.join(log_dir, f'Error_distribution_GNN_vs_TML_strip_plot'), dpi=300, bbox_inches='tight')

    plt.close()

    results_all.to_csv(os.path.join(log_dir, 'predictions_all.csv'))

    print('Plot Error_distribution_GNN_vs_ap has been saved in the directory {}'.format(log_dir))


    '''
    Error distribution plots and csv with predictions containing filtered predictions
    If the error is higher then 50%, it is not plotted 
    '''


    data = results_all.loc[abs(results_all['Error'])<50]


    violinplot(data = data, x='Fold', y='Error', hue='Method', split=True, gap=.1, inner="quart", fill=False)

    plt.xlabel('Fold Used as Test Set', fontsize=18)
    plt.ylabel('$\%top_{real}-\%top_{predicted}$', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax= plt.gca()
    ax.get_legend().remove()

    plt.savefig(os.path.join(log_dir, f'Error_distribution_GNN_vs_TML_violin_plot_filtered'), dpi=300, bbox_inches='tight')
    plt.close()

    stripplot(data=data, x="Fold", y="Error", hue="Method", size=3, dodge=True, jitter=True, marker='D', alpha=.3)

    plt.xlabel('Fold Used as Test Set', fontsize=18)
    plt.ylabel('$\%top_{real}-\%top_{predicted}$', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax= plt.gca()
    ax.get_legend().remove()
    
    plt.savefig(os.path.join(log_dir, f'Error_distribution_GNN_vs_TML_strip_plot_filtered'), dpi=300, bbox_inches='tight')

    plt.close()

    data.to_csv(os.path.join(log_dir, 'predictions_filtered.csv'))

    print('Plot Error_distribution_GNN_vs_TML has been saved in the directory {}'.format(log_dir))





if __name__ == '__main__':
    main()