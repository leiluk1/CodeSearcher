import os
import shutil
import zipfile
import re

import pandas as pd
import requests
from loguru import logger
from tqdm.auto import tqdm

DATA_RAW_FOLDER = 'data/raw/PyTorrent/'
DATA_INTERIM_FOLDER = 'data/interim/PyTorrent/'
DATA_PROCESSED_FOLDER = 'data/processed/PyTorrent/'
MODES = ['train', 'valid', 'test']

ZENODO_LINK = 'https://zenodo.org/record/4546290/files/PyTorrent_UserComments_v2.zip'


def _setup_raw_dataset():
    os.makedirs(DATA_RAW_FOLDER, exist_ok=True)
    r = requests.get(ZENODO_LINK, stream=True)
    total_size = int(r.headers['content-length'])
    chunk_size = 1024
    zip_file = f"./{DATA_RAW_FOLDER}raw_archive.zip"
    with open(zip_file, "wb") as pytorrent_archive:
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=(total_size + chunk_size - 1) // chunk_size):
            if chunk:
                pytorrent_archive.write(chunk)

    for mode in MODES:
        os.makedirs(os.path.join(DATA_RAW_FOLDER, mode), exist_ok=True)
    with zipfile.ZipFile(zip_file) as zf:
        for member in tqdm(zf.namelist(), desc='extracting files'):
            filename = os.path.basename(member)
            # skip directories
            if not filename:
                continue

            # extract file
            mode = os.path.dirname(member).split('/')[-1]

            source = zf.open(member)
            target = open(os.path.join(DATA_RAW_FOLDER, mode, filename), "wb")
            with source, target:  # close files after function exit
                shutil.copyfileobj(source, target)

    logger.info(f'Dataset downloaded and extracted to {DATA_RAW_FOLDER}')


class PyTorrentDataset:

    def __init__(self,
                 mode: str,
                 download: bool = False,
                 reload: bool = False,
                 code_tokens_cutoff_len: int = 128,
                 max_chunks=-1):
        assert mode in MODES
        self.mode = mode
        if not os.path.exists(DATA_RAW_FOLDER) or download:
            logger.info(f'Downloading the data from {ZENODO_LINK}')
            _setup_raw_dataset()
        if not os.path.exists(f'{DATA_PROCESSED_FOLDER}{mode}.jsonl') or reload:
            os.makedirs(DATA_PROCESSED_FOLDER, exist_ok=True)
            raw_dataframe = self._read_gzip_jsonl(max_chunks=max_chunks)
            logger.info(f'Raw dataframe length: {len(raw_dataframe)}')
            self.dataframe = raw_dataframe[raw_dataframe.summary.apply(lambda x: re.match('[a-zA-Z0-9_\-\s]+', x) is not None)]
            logger.info(f'Len after selecting English summaries (*, NL): {len(self.dataframe)}')
            self.dataframe = self.dataframe[self.dataframe['code_tokens'].apply(len) <= code_tokens_cutoff_len]
            self.dataframe['code_tokens'] = self.dataframe['code_tokens'].apply(lambda x: " ".join(x))
            logger.info(f'Len after dropping rows with code_tokens > {code_tokens_cutoff_len}: {len(self.dataframe)}')

            with open(f'{DATA_PROCESSED_FOLDER}{mode}.jsonl', 'w') as f:
                f.write(self.dataframe.to_json(orient='records', lines=True))
        else:
            data_path = os.path.join(DATA_PROCESSED_FOLDER, f'{self.mode}.jsonl')
            logger.info(f'Loading processed dataframe from {data_path}')
            self.dataframe = pd.read_json(data_path, lines=True)
        logger.info(f'{mode} dataset length: {len(self.dataframe)}')

    def _read_gzip_jsonl(self, max_chunks=-1):
        data = None
        archives = os.listdir(os.path.join(DATA_RAW_FOLDER, self.mode))
        if max_chunks > 0:
            archives = archives[:max_chunks]
        for archive_path in tqdm(archives, desc=f'Loading {self.mode} files into memory'):
            frame = pd.read_json(os.path.join(DATA_RAW_FOLDER, self.mode, archive_path), lines=True)
            frame = frame[['code_tokens', 'summary']]
            data = pd.concat([data, frame], ignore_index=True) if data is not None else frame

        return data

    def get_pandas(self):
        return self.dataframe


if __name__ == '__main__':
    sample_set = PyTorrentDataset('train', download=False, reload=True, max_chunks=-1)
