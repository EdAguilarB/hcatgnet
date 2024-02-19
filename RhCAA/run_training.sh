source activate ligands


python training_es.py 

python traditional_ML.py -d RhCAA/Traditional_ML/RandomForest -a rf 
python traditional_ML.py -d RhCAA/Traditional_ML/GradientBoosting -a gb 
python traditional_ML.py -d RhCAA/Traditional_ML/LinearRegression -a lr 



