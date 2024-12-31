import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm

def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()

def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(filename, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsp_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances (default 20, 50, 100,200,300,500")
    parser.add_argument("--tag", type=str, required=True, help="Tag to identify dataset")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")

    opts = parser.parse_args()

    datadir = os.path.join(opts.data_dir, 'tsp')
    os.makedirs(datadir, exist_ok=True)

 
    for tsp_size in tqdm(opts.tsp_sizes):
        filename = os.path.join(datadir, "{}{}_{}_seed{}.pkl".format(
            'tsp', tsp_size, opts.tag, opts.seed))
        
        if os.path.isfile(filename):
            if not opts.f:
                print("[{}] already exists! Done!".format(filename))
                continue
            else:
                print("Overwriting [{}]".format(filename))
                os.remove(filename)

        np.random.seed(opts.seed)
        dataset = generate_tsp_data(opts.dataset_size, tsp_size)
        save_dataset(dataset, filename)
        print("Dataset [{}] generated successfully! Done!".format(filename))

        
