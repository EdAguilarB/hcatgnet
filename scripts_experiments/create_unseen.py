import os
import pandas as pd
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from options.base_options import BaseOptions

def create_unseen_data(opt):
    """
    Function to create the unseen data for the final test
    """

    # Load the final test data
    data = pd.read_csv(f'{opt.root}/{opt.filename}', index_col=0)

    len_data = len(data)
    unseen_size = len_data * opt.unseen_ratio

    # Split the final test data into training and unseen data
    data = data.sample(frac=1, random_state=opt.global_seed)
    data_seen = data.iloc[:int(len_data - unseen_size)]
    data_unseen = data.iloc[int(len_data - unseen_size):]

    os.makedirs(f'{opt.root}/learning/raw', exist_ok=True)
    os.makedirs(f'{opt.root}/test/raw', exist_ok=True)

    # Save the training data
    data_seen.to_csv(f'{opt.root}/learning/raw/{opt.filename}')
    print('Training data has been saved in the directory {}'.format(f'{opt.root}/{opt.filename}/learning/raw'))

    # Save the unseen data
    data_unseen.to_csv(f'{opt.root}/test/raw/{opt.filename}')
    print('Unseen data has been saved in the directory {}'.format(f'{opt.root}/test/raw'))

    return None

if __name__ == '__main__':
    opt = BaseOptions().parse()
    create_unseen_data(opt)