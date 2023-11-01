import datasets

from src.datasets.PyTorrent.make_dataset import MODES as PYTORRENT_MODES
from src.datasets.PyTorrent.make_dataset import PyTorrentDataset


def preprocess_function(samples):
    # Tokenize sentences
    return None


def create_python_dataset():
    dataset = datasets.DatasetDict()
    pytorrent_dataframes = [PyTorrentDataset(mode).get_pandas() for mode in PYTORRENT_MODES]
    dataset['train'] = datasets.Dataset.from_pandas(pytorrent_dataframes[0])
    dataset['val'] = datasets.Dataset.from_pandas(pytorrent_dataframes[1])
    dataset['test'] = datasets.Dataset.from_pandas(pytorrent_dataframes[2])
