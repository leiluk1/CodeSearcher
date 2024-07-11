from datasets import load_dataset
from loguru import logger

MODES = ['train', 'validation', 'test']
LANGUAGES = ['python', 'go', 'java', 'javascript', 'ruby', 'php']
COLUMNS_RENAMING = {'func_code_string': 'code_tokens', 'func_documentation_string': 'summary'}


class CodeSearchNet:

    def __init__(self, mode, language):
        assert mode in MODES, f'Unsupported mode {mode}'
        assert language in LANGUAGES, f'Unsupported language {language}'

        self.language = language
        self.mode = mode

        self.dataframe = load_dataset("code_search_net", language)[mode].to_pandas()
        self.dataframe.drop([column for column in self.dataframe.columns if column not in COLUMNS_RENAMING],
                            axis=1,
                            inplace=True)
        self.dataframe.rename(columns=COLUMNS_RENAMING, inplace=True)

        logger.info(f'CodeSearchNet {language} {mode} dataset length: {len(self.dataframe)}')

    def get_pandas(self):
        return self.dataframe
