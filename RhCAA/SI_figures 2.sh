source activate ligands


python plot_results_all.py -t RhCASA/Traditional_ML/RandomForest -l SI/RhCASA/Results_RandomForest
python plot_results_all.py -t RhCASA/Traditional_ML/LinearRegression -l SI/RhCASA/Results_LinearRegression

python final_test_traditional_ML.py -m RhCASA/Traditional_ML/RandomForest -d SI/bicyclic_test/RandomForest
python final_test_traditional_ML.py -m RhCASA/Traditional_ML/LinearRegression -d SI/bicyclic_test/LinearRegression

python plot_results_all.py -g bicyclic_test/GNN -t SI/bicyclic_test/RandomForest -l SI/bicyclic_test/RandomForest/Results
python plot_results_all.py -g bicyclic_test/GNN -t SI/bicyclic_test/LinearRegression -l SI/bicyclic_test/LinearRegression/Results



