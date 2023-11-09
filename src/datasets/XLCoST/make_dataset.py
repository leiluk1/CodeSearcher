import os.path
import zipfile

import gdown
import pandas as pd
from loguru import logger

DATA_RAW_FOLDER = 'data/raw/XLCoST_data/'
DATA_ID = "1tZfsYQgWmc2gG340ru5VbrZ5aLIZ41_6"
MODES = ['train', 'valid', 'test']
LANGUAGES = ['C', 'C#', 'C++', 'Java', 'Javascript', 'PHP', 'Python']


def _setup_dataset():
    logger.info(f"Downloading XLCoST from https://drive.google.com/file/d/{DATA_ID}")
    zip_path = f"{DATA_RAW_FOLDER}XLCoST_data.zip"
    gdown.download(id=DATA_ID, output=zip_path, quiet=False)
    logger.info('Extracting files')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data/raw')
    logger.info('Extraction complete')


def _code_tokens_to_str(tokens: list):
    indent_level = 0
    result = ''
    line = ''
    for token in tokens:
        if token == 'INDENT':
            indent_level += 1
        elif token == 'DEDENT':
            indent_level -= 1
        elif token == 'NEW_LINE':
            result += ('\n' + '\t' * indent_level + line)
            line = ''
        else:
            line += token + " "
    return result[1:]


class XLCoSTDataset:

    def __init__(self, mode, language, generation: bool = False, level: str = 'snippet', download: bool = False,
                 min_tokens: int = 3):
        assert mode in MODES, f'Unsupported mode {mode}'
        assert language in LANGUAGES, f'Unsupported language {language}'

        self.language = language

        if not os.path.exists(DATA_RAW_FOLDER) or download:
            _setup_dataset()

        self.mode = mode
        self.generation = generation
        self.level = level
        self.min_tokens = min_tokens

        if generation:
            self.dataframe = self._load_generation_dataframe()
        else:
            self.dataframe = self._load_search_dataframe()
        logger.info(f'XLCoST {language} {mode} generation={generation} dataset length: {len(self.dataframe)}')

    def _load_generation_dataframe(self):
        from_data_root_path = 'generation/pair_data_tok_full_desc'
        language_folder = f'{self.language}-desc'
        data_path = os.path.join(DATA_RAW_FOLDER, from_data_root_path, language_folder)
        with open(os.path.join(data_path, f'{self.mode}-{self.language}-desc-tok.txt'), 'r') as summary_file:
            summary = [line.split() for line in summary_file.readlines()]
        with open(os.path.join(data_path, f'{self.mode}-{self.language}-desc-tok.txt'), 'r') as code_file:
            code_tokens = [line.split() for line in code_file.readlines()]
        dataframe = pd.DataFrame({'summary': summary, 'code_tokens': code_tokens})

        return self._process_dataframe(dataframe)

    def _load_search_dataframe(self):
        from_data_root_path = 'retrieval/nl2code_search'
        language_folder = f'{self.level}_level/{self.language}'
        data_path = os.path.join(DATA_RAW_FOLDER, from_data_root_path, language_folder)

        data_path = os.path.join(data_path, f'{self.mode}.jsonl')
        logger.info(f'Loading dataframe from {data_path}')
        dataframe = pd.read_json(data_path, lines=True)[['docstring_tokens', 'code_tokens']]
        dataframe = dataframe.rename(columns={'docstring_tokens': 'summary'})

        return self._process_dataframe(dataframe)

    def _process_dataframe(self, dataframe):
        dataframe_trunc = dataframe[dataframe.summary.apply(len) >= self.min_tokens]
        dataframe_trunc = dataframe_trunc[dataframe_trunc.code_tokens.apply(len) >= self.min_tokens]

        dataframe_trunc['code_tokens'] = dataframe_trunc['code_tokens'].apply(_code_tokens_to_str)
        dataframe_trunc['summary'] = dataframe_trunc['summary'].apply(lambda x: " ".join(x))
        return dataframe_trunc

    def get_pandas(self):
        return self.dataframe


if __name__ == "__main__":
    XLCoSTDataset(mode='train', language='Python', download=False, level='program')
