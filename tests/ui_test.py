from transformers import AutoTokenizer, AutoModel
import torch
from peft import PeftModel, PeftConfig
import numpy as np
from annoy import AnnoyIndex

import sys
sys.path.append('../')

from CodeSearcher.src.datasets.make_datasets import create_python_dataset, create_java_dataset, \
    create_csharp_dataset, create_sql_dataset, create_cpp_dataset

DATASET_MAP = {'Python': create_python_dataset, 'Java': create_java_dataset,
               'Csharp': create_csharp_dataset, 'SQL': create_sql_dataset, 'C++': create_cpp_dataset}



def find_nearest(embedding, index, id_map, k=1):
    nearest_ids = index.get_nns_by_vector(embedding, k)
    nearest_codes = [id_map[i] for i in nearest_ids]
    return nearest_codes


def _setup_seq2seq_dataset(raw_dataset, tokenizer, model_max_src_length, model_max_tgt_length, num_virtual_tokens):
    def preprocess_function(examples):
        model_inputs = tokenizer(examples['code_tokens'], max_length=model_max_src_length - num_virtual_tokens,
                                 padding='max_length', truncation=True)
        summary = tokenizer(examples['summary'], max_length=model_max_tgt_length - num_virtual_tokens,
                            padding='max_length', truncation=True)
        summary[summary == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = summary["input_ids"].copy()
        return model_inputs

    tokenized_dataset = raw_dataset.map(preprocess_function,
                                        batched=True,
                                        remove_columns=raw_dataset["train"].column_names,
                                        desc='Tokenizing dataset')
    return tokenized_dataset


def get_decoded_text_from_model(checkpoint_path, input_text, language='Python'):
    device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device_type)
    config = PeftConfig.from_pretrained(checkpoint_path)
    # Load the model and tokenizer
    model = AutoModel.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, checkpoint_path)
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    # Encode the input text
    encoded_input = tokenizer.encode_plus(input_text, return_tensors="pt", padding='max_length', max_length=54, truncation=True)
    input_ids = encoded_input['input_ids'].to(next(model.parameters()).device)

    is_encoder_only = 'seq2seq' not in str(type(model)).lower()
    if is_encoder_only:
        output = model(**encoded_input)
    else:
        output = model.encoder(encoded_input['input_ids']).last_hidden_state[:, 0, :]

    raw_dataset = DATASET_MAP[language]  

    # print('raw_dataset', raw_dataset.head(5))  

    # Assume embeddings is a list of your code embeddings and codes is a list of the corresponding codes
    embeddings = ...
    codes = ...

    # Build the Annoy index
    index = AnnoyIndex(len(embeddings[0]), 'angular')  # Length of item vector that will be indexed
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)
    index.build(10)  # 10 trees

    # Build the id map
    id_map = {i: code for i, code in enumerate(codes)}

    # Find the nearest code
    nearest_codes = find_nearest(output, index, id_map, k=1)
    
    return nearest_codes

checkpoint_path = "checkpoints/codet5p-220m-seq2seq/prefix-python/"
input_text = "print hello world"
decoded_text = get_decoded_text_from_model(checkpoint_path, input_text)
