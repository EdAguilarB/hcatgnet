source activate ligands


python plot_results_all.py -t RhCAA/Traditional_ML/RandomForest -l SI/RhCAA/Results_RandomForest
python plot_results_all.py -t RhCAA/Traditional_ML/LinearRegression -l SI/RhCAA/Results_LinearRegression

python final_test_traditional_ML.py -m RhCAA/Traditional_ML/RandomForest -d SI/final_test/RandomForest
python final_test_traditional_ML.py -m RhCAA/Traditional_ML/LinearRegression -d SI/final_test/LinearRegression

python plot_results_all.py -g final_test/GNN -t SI/final_test/RandomForest -l SI/final_test/RandomForest/Results
python plot_results_all.py -g final_test/GNN -t SI/final_test/LinearRegression -l SI/final_test/LinearRegression/Results



