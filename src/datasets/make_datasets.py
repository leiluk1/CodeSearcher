import datasets
import pandas as pd

from src.datasets.CoDesc.make_dataset import CoDescDataset
from src.datasets.CoDesc.make_dataset import MODES as CODESC_MODES
from src.datasets.PyTorrent.make_dataset import MODES as PYTORRENT_MODES
from src.datasets.PyTorrent.make_dataset import PyTorrentDataset
from src.datasets.StaQC.make_dataset import MODES as STAQC_MODES
from src.datasets.StaQC.make_dataset import StaQCDataset
from src.datasets.XLCoST.make_dataset import MODES as XLCOST_MODES
from src.datasets.XLCoST.make_dataset import XLCoSTDataset


def create_python_dataset(max_length):  # subset: str = 'all'
    dataset = datasets.DatasetDict()
    pytorrent_dataframes = [PyTorrentDataset(mode, code_tokens_cutoff_len=max_length).get_pandas() for mode in
                            PYTORRENT_MODES]

    xlcost_dataframes = [XLCoSTDataset(mode=mode, language='Python').get_pandas() for mode in XLCOST_MODES]

    staqc_dataframes = [StaQCDataset(mode=mode, language='Python').get_pandas() for mode in STAQC_MODES]

    dataframes = [
        pd.concat(
            [pytorrent_dataframes[i], xlcost_dataframes[i], staqc_dataframes[i]],
            ignore_index=True
        )
        for i in range(3)
    ]

    dataset['train'] = datasets.Dataset.from_pandas(dataframes[0])
    dataset['val'] = datasets.Dataset.from_pandas(dataframes[1])
    dataset['test'] = datasets.Dataset.from_pandas(dataframes[2])
    return dataset


def create_java_dataset(**kwargs):
    dataset = datasets.DatasetDict()

    xlcost_dataframes = [XLCoSTDataset(mode=mode, language='Java').get_pandas() for mode in XLCOST_MODES]

    codesc_dataframes = [CoDescDataset(mode=mode).get_pandas() for mode in CODESC_MODES]

    dataframes = [
        pd.concat(
            [xlcost_dataframes[i], codesc_dataframes[i]],
            ignore_index=True
        )
        for i in range(3)
    ]

    dataset['train'] = datasets.Dataset.from_pandas(dataframes[0])
    dataset['val'] = datasets.Dataset.from_pandas(dataframes[1])
    dataset['test'] = datasets.Dataset.from_pandas(dataframes[2])

    return dataset
