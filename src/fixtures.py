from src.datasets.make_datasets import create_python_dataset, create_java_dataset, \
    create_csharp_dataset, create_sql_dataset, create_cpp_dataset

DATASET_MAP = {'Python': create_python_dataset, 'Java': create_java_dataset,
               'Csharp': create_csharp_dataset, 'SQL': create_sql_dataset, 'C++': create_cpp_dataset}