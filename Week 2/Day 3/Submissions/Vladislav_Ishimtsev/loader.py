import os
import pandas as pd
import tarfile
import urllib.request

from glob import glob
from utils import Logger

class DataLoader:
    def __init__(self, data_dir='data', json_dir='flightjson', dataset_name='NYC Flights', n_rows=-1, verbose=True,
                 url="https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz"):
        self.data_dir = data_dir
        self.json_dir = os.path.join(self.data_dir, json_dir)
        self.tar_name = os.path.join(self.data_dir, 'archive.tar.gz')

        self.dataset_name = dataset_name
        self.download_url = url
        self.n_rows = n_rows

        self.step_download = False
        self.step_extract = False
        self.step_to_json = False

        self.logger = Logger(verbose)
        os.makedirs(data_dir, exist_ok=True)

    def setting_up(self):
        self.logger.print_("Setting up data directory\n-------------------------")

        self.download()
        self.extract()
        self.to_json()

        self.logger.print_('Finished!')

    def download(self):
        self.logger.print_('Downloading ' + self.dataset_name + ' dataset...', end='')

        if self.step_download:
            self.logger.print_('Using cached ' + self.tar_name + '...', end='')
        else:
            urllib.request.urlretrieve(self.download_url, self.tar_name)

        self.logger.print_('Done')
        self.step_download = True

    def extract(self):
        if not self.step_download:
            raise Exception('Error: should download file first!')

        self.logger.print_('Extracting ' + self.dataset_name + ' data...', end='')

        if self.step_extract:
            self.logger.print_('Using cached ' + self.tar_name + '...', end='')
        else:
            with tarfile.open(self.tar_name, mode='r:gz') as f:
                f.extractall(self.data_dir)

        self.logger.print_('Done')
        self.step_extract = True

    def to_json(self):
        if not self.step_extract:
            raise Exception('Error: should extract files first!')

        self.logger.print_('Creating JSON data from ' + self.dataset_name + ' data...', end='')
        os.makedirs(self.json_dir, exist_ok=True)

        if self.step_to_json:
            self.logger.print_('Using cached ' + self.json_dir + '...', end='')
        else:
            for path in glob(os.path.join(self.data_dir, '*', '*.csv')):
                prefix = os.path.splitext(os.path.basename(path))[0]

                # Just take the first 10000 rows for the demo
                if self.n_rows < 0:
                    df = pd.read_csv(path)
                else:
                    df = pd.read_csv(path, nrows=self.n_rows)

                df.to_json(os.path.join(self.json_dir, prefix + '.json'), orient='records', lines=True)

        self.logger.print_('Done')
        self.step_to_json = True
