import os
import sys

import torch
from fire import Fire
from loguru import logger
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, DataCollatorForSeq2Seq


def evaluation(model, test_loader, device, desc='Test set MRR = '):
    from src.metrics import mrr

    model.eval()
    test_text_embeddings = []
    test_code_embeddings = []
    is_encoder_only = '220m-bimodal' not in str(type(model))  # Rethink this later
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
    similarity_matrix = torch.cat(test_text_embeddings) @ torch.cat(test_code_embeddings).T
    test_MRR = mrr(similarity_matrix)
    logger.info(f'{desc}{test_MRR}')


def eval_model(tuned_ckpt_path,
               language,
               num_virtual_tokens: int,
               model_max_src_length: int = 128,
               model_max_tgt_length: int = 128,
               batch_size: int = 16,
               device_type: str = 'cuda:0'):
    device = torch.device(device_type)
    config = PeftConfig.from_pretrained(tuned_ckpt_path)

    model = AutoModel.from_pretrained(config.base_model_name_or_path, device_map={"": 0}, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, tuned_ckpt_path)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    raw_dataset = DATASET_MAP[language](max_length=model_max_src_length)

    tokenized_dataset = _setup_seq2seq_dataset(raw_dataset, tokenizer, model_max_src_length, model_max_tgt_length,
                                               num_virtual_tokens)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=config.base_model_name_or_path)

    test_loader = DataLoader(tokenized_dataset['test'], batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    evaluation(model, test_loader, device, desc=f'Test MRR for {language} = ')


if __name__ == '__main__':
    sys.path.append(os.getcwd())

    from src.fixtures import DATASET_MAP
    from src.models.train import _setup_seq2seq_dataset

    Fire(eval_model)
