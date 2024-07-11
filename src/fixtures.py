from functools import partial

from src.datasets.make_datasets import create_python_dataset, create_java_dataset, \
    create_csharp_dataset, create_sql_dataset, create_cpp_dataset, create_csn_dataset

DATASET_MAP = {'Python': create_python_dataset,
               'Java': create_java_dataset,
               'Csharp': create_csharp_dataset,
               'SQL': create_sql_dataset,
               'C++': create_cpp_dataset,
               'CSN_ruby': partial(create_csn_dataset, language='ruby'),
               'CSN_javascript': partial(create_csn_dataset, language='javascript'),
               'CSN_java': partial(create_csn_dataset, language='java'),
               'CSN_go': partial(create_csn_dataset, language='go'),
               'CSN_php': partial(create_csn_dataset, language='php'),
               'CSN_python': partial(create_csn_dataset, language='python')
               }
