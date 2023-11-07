import gzip
import os.path
from ast import literal_eval
from random import sample

import gdown
import pandas as pd
import py7zr
from loguru import logger

DATA_RAW_FOLDER = 'data/raw/CoDesc_data/'
URL = "https://drive.google.com/uc?id=1NKHb_mCH345NXcMFUBw5SxOgki8N5wsO"
MODES = ['train', 'test', 'val']


def retrieve_json(option='train'):
    json_list = []
    path = DATA_RAW_FOLDER + f"csn_preprocessed_data_balanced_partition/java/final/jsonl/{option}/"
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if file_name.endswith('.jsonl.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8') as json_file:
                lines = json_file.readlines()
                json_list.extend(lines)
    return json_list


def _setup_dataset():
    logger.info(f"Downloading CoDesc from {URL}")
    archive_path = f"{DATA_RAW_FOLDER}archive.7z"
    gdown.download(URL, archive_path, quiet=False)
    logger.info('Extracting files')
    with py7zr.SevenZipFile(archive_path, mode='r') as z:
        z.extractall(DATA_RAW_FOLDER)
    logger.info('Extraction complete')


class CoDescDataset:
    def __init__(self, mode='train', download=False):
        assert mode in MODES, f'Unsupported mode {mode}'

        self._load_dataframe(download)
        self.mode = mode
        self.json_list = retrieve_json(mode)
        self.df = self._process_dataframe()

    def _load_dataframe(self, download=False):
        if not os.path.exists(DATA_RAW_FOLDER) or download:
            _setup_dataset()

    def _process_dataframe(self):
        sample_size = len(self.json_list)
        dataframe = pd.DataFrame([literal_eval(line) for line in sample(self.json_list, sample_size)]).set_index('id')
        dataframe = dataframe[['docstring_tokens', 'code_tokens']]

        # rename columns
        dataframe = dataframe.rename(columns={'docstring_tokens': 'summary'})
        logger.info(f"Loaded {len(dataframe)} {self.mode} examples")
        return dataframe

    def get_pandas(self):
        return self.df


if __name__ == '__main__':
    dataset = CoDescDataset(mode='train')
