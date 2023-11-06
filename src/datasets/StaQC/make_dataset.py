from datasets import load_dataset
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


# DATA_RAW_FOLDER = 'data/raw/StaQC/'
MODES = ['train', 'test', 'val']
LANGUAGES = ['Python', 'SQL']
SUBSET_MODES = ['man_python', 'sca_python', 'man_sql', 'sca_sql']


def _parse_df(man_df):
    rows = []
    for _, row in man_df.iterrows():
        snippets = row['snippet']
        collected_snippets = []
        for i, snippet in enumerate(snippets['text']):
            if snippets['is_sda'][i]:
                collected_snippets.append(snippet)
        for snippet in collected_snippets:
            new_row = pd.Series(row[['question_id', 'question']].tolist() + [snippet],
                                index=['question_id', 'question', 'snippet'])
            rows.append(new_row)

    parsed_df = pd.DataFrame(rows, columns=['question_id', 'question', 'snippet'])
    return parsed_df 


def _setup_dataset(language: str = 'Python'):
    language = language.lower()
    man_df = load_dataset("koutch/staqc", f'man_{language}')['train'].to_pandas()
    sca_df = load_dataset("koutch/staqc", f'sca_{language}')['train'].to_pandas()
    parsed_man_df = _parse_df(man_df)
    df = pd.concat([parsed_man_df, sca_df])
    df.drop('question_id', axis=1, inplace=True)
    
    logger.info('Loading complete')
    return df


def _split_dataset(df, mode):
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42)
    if mode == 'train':
        return df_train
    
    elif mode == 'test':
        return df_test
    
    elif mode == 'val':
        return df_val


class StaQCDataset:

    def __init__(self, mode, language, min_tokens: int = 3):
        assert mode in MODES, f'Unsupported mode {mode}'
        assert language in LANGUAGES, f'Unsupported language {language}'

        self.language = language
        self.mode = mode
        self.min_tokens = min_tokens
        
        self.dataframe = self._load_dataframe()
        # print(self.dataframe.sample(5))
        
        logger.info(f'StaQC {language} {mode} dataset length: {len(self.dataframe)}')

    def _load_dataframe(self):
        setup_dataframe = _setup_dataset(self.language) 
        dataframe_mode = _split_dataset(setup_dataframe, self.mode)
        dataframe = pd.DataFrame({'summary': dataframe_mode.question.values, 
                                  'code_tokens': dataframe_mode.snippet.values})

        return self._process_dataframe(dataframe)

    def _process_dataframe(self, dataframe):
        dataframe_trunc = dataframe[dataframe.summary.apply(len) >= self.min_tokens]
        dataframe_trunc = dataframe_trunc[dataframe_trunc.code_tokens.apply(len) >= self.min_tokens]
        return dataframe_trunc

    def get_pandas(self):
        return self.dataframe


if __name__ == "__main__":
   StaQCDataset(mode='train', language='SQL')
