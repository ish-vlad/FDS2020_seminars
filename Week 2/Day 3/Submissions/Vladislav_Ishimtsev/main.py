import argparse

from dask.distributed import Client, LocalCluster
from model import ScikitModel, DaskModel


def test_model(clf, data_dir, param_grid):
    clf.load_data(data_dir)
    clf.optimize(param_grid)
    clf.train()
    clf.evaluate()


def main(args):
    verbose = args.v
    n_jobs = args.n
    data_dir = args.d
    param_grid = {
        'max_depth': [1, 10, None],
        'n_estimators': [10, 50, 100],
    }

    # Scikit model
    clf = ScikitModel(n_jobs=n_jobs, verbose=verbose)
    test_model(clf, data_dir, param_grid)
    print(clf)

    # Dask model
    with Client(threads_per_worker=4, n_workers=n_jobs, memory_limit='128GB') as client:
        clf = DaskModel(client=client, n_jobs=n_jobs, verbose=verbose)
        test_model(clf, data_dir, param_grid)
        print(clf)


def parse_args():
    parser = argparse.ArgumentParser(description='Run model testing')

    parser.add_argument('-d', default='./data', help='Directory with data')
    parser.add_argument('-n', default=20, type=int, help='Number of jobs')
    parser.add_argument('-v', action='store_true', default=False, help='Verbose. Do we need to activate all prints?')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args.verbose)
