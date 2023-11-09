import gc
import os
import random
import sys

import evaluate
import numpy as np
import torch
from fire import Fire
from loguru import logger
from peft import PrefixTuningConfig, TaskType, get_peft_model, PromptEncoderConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, AutoModelForSeq2SeqLM, \
    AutoTokenizer, AutoModel


def _setup_seq2seq_model(model_checkpoint, num_virtual_tokens, device):
    model = AutoModel.from_pretrained(model_checkpoint, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)

    peft_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        num_virtual_tokens=num_virtual_tokens
    )
    for param in model.parameters():
        param.requires_grad = False

    model = get_peft_model(model, peft_config)

    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )
    model.to(device)

    return model, tokenizer


def _setup_embeddings_model(model_checkpoint, num_virtual_tokens, device):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_checkpoint, trust_remote_code=True)

    peft_config = PromptEncoderConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        num_virtual_tokens=num_virtual_tokens,
        inference_mode=False
    )
    for param in model.parameters():
        param.requires_grad = False

    model = get_peft_model(model, peft_config)

    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )
    model.to(device)

    with torch.inference_mode():
        inputs = tokenizer.encode("def print_hello_world():\tprint('Hello World!')", return_tensors="pt").to(device)
        embedding = model(inputs)
        assert embedding.shape == (1, 256) and np.allclose(embedding.norm().item(), 1.0, atol=1e-7), \
            'Error while creating the model'

    return model, tokenizer


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


def train_seq2seq(output_dir: str,
                  language: str,
                  epochs: int,
                  num_virtual_tokens,
                  model_max_src_length: int = 320,
                  model_max_tgt_length: int = 128,
                  train_batch_size: int = 32,
                  eval_batch_size: int = 16,
                  gradient_accumulation_steps: int = 4,
                  warmup_steps: int = 200,
                  fp16: bool = False,
                  device_type: str = 'cuda:0',
                  model_checkpoint: str = 'Salesforce/codet5p-220m-bimodal'
                  ):
    device = torch.device(device_type)
    model, tokenizer = _setup_seq2seq_model(model_checkpoint, num_virtual_tokens, device)

    raw_dataset = DATASET_MAP[language](model_max_src_length)

    tokenized_dataset = _setup_seq2seq_dataset(raw_dataset, tokenizer, model_max_src_length, model_max_tgt_length,
                                               num_virtual_tokens)

    # This code is taken from https://huggingface.co/docs/transformers/tasks/summarization
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    arguments = Seq2SeqTrainingArguments(
        output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=fp16,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        arguments,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, 'best_ckpt'))


def train_embeddings(output_dir: str,
                     epochs: int,
                     language,
                     num_virtual_tokens: int = 20,
                     model_max_src_length: int = 320,
                     model_max_tgt_length: int = 128,
                     train_batch_size: int = 32,
                     eval_batch_size: int = 16,
                     gradient_accumulation_steps: int = 4,
                     label_smoothing: float = 0,
                     device_type: str = 'cuda:0',
                     model_checkpoint: str = 'Salesforce/codet5p-110m-embedding',
                     ):
    # Future refactoring plan: to avoid duplicate code, abstract training from model obtaining ?
    #   (get model) -> (choose training type) -> ...
    device = torch.device(device_type)
    embeddings_model, tokenizer = _setup_embeddings_model(model_checkpoint, num_virtual_tokens, device)
    peft_model_id = os.path.join(output_dir, model_checkpoint)

    raw_dataset = DATASET_MAP[language](model_max_src_length)

    pairs_dataset = _setup_seq2seq_dataset(raw_dataset, tokenizer, model_max_src_length, model_max_tgt_length,
                                           num_virtual_tokens)
    train_dataset_len, val_dataset_len, test_dataset_len = len(pairs_dataset['train']), len(pairs_dataset['val']), \
        len(pairs_dataset['test'])
    logger.info(f'Dataset created: {train_dataset_len} training samples, {val_dataset_len} validation samples,'
                f' {test_dataset_len} testing samples')

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_checkpoint)  # We still need seq2seq dataset

    train_loader = DataLoader(pairs_dataset['train'], batch_size=train_batch_size, shuffle=True,
                              collate_fn=data_collator)
    val_loader = DataLoader(pairs_dataset['val'], batch_size=eval_batch_size, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(pairs_dataset['test'], batch_size=eval_batch_size, shuffle=False, collate_fn=data_collator)

    emb_loss = TextCodeContrastiveLoss(smooth=label_smoothing)
    optimizer = torch.optim.AdamW(embeddings_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    logger.info('Training started')
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        embeddings_model.train()
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train epoch', position=0):
            batch.to(device)
            text_batch = batch.pop("labels")
            code_embeddings = embeddings_model(**batch)
            text_embeddings = embeddings_model(text_batch)

            loss = emb_loss(text_batch=text_embeddings, code_batch=code_embeddings)
            loss.backward()
            train_loss += loss.item()

            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        logger.info(f"Training loss of epoch {epoch}: {train_loss / len(train_loader)}")

        val_loss = 0
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='Val epoch', position=0):
            batch.to(device)
            text_batch = batch.pop("labels")

            with torch.no_grad():
                code_embeddings = embeddings_model(**batch)
                text_embeddings = embeddings_model(text_batch)
                loss = emb_loss(text_batch=text_embeddings, code_batch=code_embeddings)
            val_loss += loss.item()
        logger.info(f"Validation loss of epoch {epoch}: {val_loss / len(val_loader)}")
        if val_loss / len(val_loader) < best_val_loss:
            best_val_loss = val_loss / len(val_loader)
            embeddings_model.save_pretrained(peft_model_id)

    # config = PeftConfig.from_pretrained(peft_model_id)
    #
    # model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map={"": 0})
    # model = PeftModel.from_pretrained(model, peft_model_id)
    # model.eval()


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    torch.manual_seed(0)
    random.seed(0)

    from src.datasets.make_datasets import create_python_dataset, create_java_dataset
    from src.losses import TextCodeContrastiveLoss

    DATASET_MAP = {'Python': create_python_dataset, 'Java': create_java_dataset, 'C#': None, 'SQL': None}

    Fire(
        {
            'seq2seq': train_seq2seq,
            'embeddings': train_embeddings,
        }
    )
