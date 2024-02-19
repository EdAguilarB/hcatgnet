# The RhCAA reaction

The directory contains all the code necessary to replicate the resuts obtained for the RhCAA reactions.

The directories contain the following information:
1. Data: contains the script to create the folds for nested cross validation and the datasets used for the study.
2. final_test: contains the results obtained for the GNN and TMl methods to predict on the truly unseen test set.
3. RhCAA: contains the results of the nested cross validation applied to the learning dataset.
4. SI: contains the results of linear regression and random forest predicting both datasets.
5. Utils: contains functions necessary to implement the code.

The files contain the following:
1. explain_models: presents the code to explain the models and create plots of molecules showing heatmaps of what atoms contribute to the reactionn outcome.
2. final_test_traditional_ML.py has the code to predict the final_test set using the TML algorithms.
3. final_test.py: contains the code to predict the final_test set using the GNN models.
4. plot_results_alll.py: contains the code to create figures showing the metrics per test fold and error distributions of models.
5. traditional_ML.py: contains the code to run the training of the TML algorithms using the learning dataset.
6. training_es.py: contains the code to train the GNN models using the learning dataset.

The .sh scripts contain the exact commands to run in order to replicate the results reported.
To run all the experiments with the default parameters, run the script run_all_experiments.sh

