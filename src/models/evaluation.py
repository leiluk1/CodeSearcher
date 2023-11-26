import os
import sys

import numpy as np
import torch
from fire import Fire
from loguru import logger
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, DataCollatorForSeq2Seq


def evaluation(model, test_loader, device, desc='Test set MRR = ', max_batches=float('inf')):
    """

    :param model: seq2seq or encoder model
    :param test_loader: test set dataloader
    :param device: device, on which the model is stored
    :param desc: str, description of logging MRR
    :param max_batches: Max batches to consider for testing
    :return: None, log MRR to console
    """
    from src.metrics import mrr

    model.eval()
    test_text_embeddings = []
    test_code_embeddings = []
    model_type = str(type(model)).lower()
    is_encoder_only = '220m-bimodal' not in model_type and 'seq2seq' not in model_type
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing the model', position=0):
        batch.to(device)
        text_batch = batch.pop("labels")

        with torch.no_grad():
            if is_encoder_only:
                test_code_embeddings.append(model(**batch).cpu())
                test_text_embeddings.append(model(text_batch, attention_mask=(text_batch != 0).int()).cpu())
            else:
                test_code_embeddings.append(model.encoder(batch['input_ids']).last_hidden_state[:, 0, :].cpu())
                test_text_embeddings.append(model.encoder(text_batch).last_hidden_state[:, 0, :].cpu())
        if len(test_code_embeddings) == max_batches:
            break
    test_text_embeddings = np.concatenate(test_text_embeddings, 0)
    test_code_embeddings = np.concatenate(test_code_embeddings, 0)
    similarity_matrix = test_text_embeddings @ test_code_embeddings.T
    del test_text_embeddings
    del test_code_embeddings
    test_MRR = mrr(similarity_matrix)
    logger.info(f'{desc}{test_MRR}')


def eval_peft_model(tuned_ckpt_path: str,
                    language: str,
                    model_max_src_length: int = 128,
                    model_max_tgt_length: int = 128,
                    batch_size: int = 16,
                    max_batches: float = float('inf'),
                    device_type: str = 'cuda:0'):
    """

    :param tuned_ckpt_path: Local checkpoint path (for instance, "checkpoints/codet5p-110m-embedding/adalora-cpp")
    :param language: str, language to test the model on
    :param model_max_src_length: Max tokens in text encoding
    :param model_max_tgt_length: max tokens in code encoding
    :param batch_size: testing dataloader's batch size
    :param max_batches: Max number of batches on which retrieval metric will be evaluated
        (effectively the testing will happen on min(num_test_samples_for_language, batch_size * max_batches)
    :param device_type: torch.device(device_type) will contain the model
    :return: None, print MRR to the std output
    """
    device = torch.device(device_type)
    config = PeftConfig.from_pretrained(tuned_ckpt_path)

    try:
        num_virtual_tokens = config.num_virtual_tokens
    except AttributeError:
        num_virtual_tokens = 0

    model = AutoModel.from_pretrained(config.base_model_name_or_path, device_map={"": 0}, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, tuned_ckpt_path)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    raw_dataset = DATASET_MAP[language](max_length=model_max_src_length)

    tokenized_dataset = _setup_seq2seq_dataset(raw_dataset, tokenizer, model_max_src_length, model_max_tgt_length,
                                               num_virtual_tokens)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=config.base_model_name_or_path)

    test_loader = DataLoader(tokenized_dataset['test'], batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    evaluation(model, test_loader, device, desc=f'Test MRR for {language} = ', max_batches=max_batches)


def eval_base_model(model_name: str,
                    language: str,
                    model_max_src_length: int = 128,
                    model_max_tgt_length: int = 128,
                    batch_size: int = 16,
                    max_batches: float = float('inf'),
                    device_type: str = 'cuda:0'):
    """

    :param model_name: Model checkpoint on huggingface hub, for instance, "Salesforce/codet5p-110m-embedding"
    :param language: str, language to test the model on
    :param model_max_src_length: Max tokens in text encoding
    :param model_max_tgt_length: max tokens in code encoding
    :param batch_size: testing dataloader's batch size
    :param max_batches: Max number of batches on which retrieval metric will be evaluated
        (effectively the testing will happen on min(num_test_samples_for_language, batch_size * max_batches)
    :param device_type: torch.device(device_type) will contain the model
    :return: None, print MRR to the std output
    """
    device = torch.device(device_type)

    model = AutoModel.from_pretrained(model_name, device_map={"": 0}, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    raw_dataset = DATASET_MAP[language](max_length=model_max_src_length)

    tokenized_dataset = _setup_seq2seq_dataset(raw_dataset, tokenizer, model_max_src_length, model_max_tgt_length, 0)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

    test_loader = DataLoader(tokenized_dataset['test'], batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    evaluation(model, test_loader, device, desc=f'Test 0-shot MRR for {language} = ', max_batches=max_batches)


if __name__ == '__main__':
    sys.path.append(os.getcwd())

    from src.fixtures import DATASET_MAP
    from src.models.train import _setup_seq2seq_dataset

    Fire({
        'base': eval_base_model,
        'peft': eval_peft_model,
    })
