import pandas as pd
import os
import argparse
from sklearn.model_selection import StratifiedKFold

PARSER = argparse.ArgumentParser(description="Create random folds for cross validation and inner cross validation\
                                 splitting strategy for GNN approach and atomistic potential training and evaluation process. \
                                 This ensures that both models are trained using the exact same datapoints.")

PARSER.add_argument("-d", "--dir", dest="d", type=str, default="./Monocyclic",
                    help="Name of the directory where the csv is stored and where the new one will be stored.")
PARSER.add_argument("-s", "--seed", type=int, dest="s", default=20232023,
                    help="Seed for random number generator.")
PARSER.add_argument("-f", "--folds", type=int, dest="f", default=10,
                    help="Number of folds to create.")
ARGS = PARSER.parse_args()



def create_folds(num_folds, df):
    """
    splits a dataset in a given quantity of folds

    Args:
    num_folds = number of folds to create
    df = dataframe to be splited

    Returns:
    dataset with new "folds" and "mini_folds" column with information of fold for each datapoint
    """

    # Calculate the number of data points in each fold
    fold_size = len(df) // num_folds
    remainder = len(df) % num_folds

    # Create a 'fold' column to store fold assignments
    fold_column = []

    # Assign folds
    for fold in range(1, num_folds + 1):
        fold_count = fold_size
        if fold <= remainder:
            fold_count += 1
        fold_column.extend([fold] * fold_count)

    # Assign the 'fold' column to the DataFrame
    df['fold'] = fold_column

    return df



def main():

    ligands = pd.read_csv(os.path.join(ARGS.d, 'monocyclic.csv'), index_col=0)

    ligands['category'] = ligands['%top'].apply(lambda m: 0 if m < 50 else 1)

    folds = StratifiedKFold(n_splits = ARGS.f, shuffle = True, random_state=ARGS.s)

    test_idx = []

    for _, test in folds.split(ligands['ER1'], ligands['category']):
        test_idx.append(test)

    index_dict = {index: list_num for list_num, index_list in enumerate(test_idx) for index in index_list}

    ligands['fold'] = ligands.index.map(index_dict)


    ligands.to_csv(os.path.join(ARGS.d, 'raw', f'monocyclic_folds.csv'))

    save_dir = os.path.join(os.getcwd(), 'raw') if ARGS.d == '.' else os.path.join(ARGS.d, f'{ARGS.f}_folds', 'raw')

    print('monocyclic_folds.csv file was saved in {}'.format(save_dir))



if __name__ == "__main__":  
    main()  