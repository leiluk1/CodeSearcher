import os
import random
import sys
from typing import Optional, List, Dict

import evaluate
import numpy as np
import torch
from fire import Fire
from loguru import logger
from peft import PrefixTuningConfig, TaskType, get_peft_model, LoraConfig, PromptTuningConfig, PromptEncoderConfig
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, AutoModelForSeq2SeqLM, \
    AutoTokenizer, AutoModel, Trainer


def _setup_seq2seq_model(model_checkpoint, num_virtual_tokens, device):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

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
        labels = tokenizer(examples['summary'], max_length=model_max_tgt_length - num_virtual_tokens,
                           padding='max_length', truncation=True)
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels["input_ids"].copy()
        return model_inputs

    tokenized_dataset = raw_dataset.map(preprocess_function,
                                        batched=True,
                                        remove_columns=raw_dataset["train"].column_names,
                                        load_from_cache_file=False,
                                        num_proc=-1)
    return tokenized_dataset


def _setup_apn_dataset(raw_dataset, negatives_strategy, tokenizer,
                       model_max_src_length, model_max_tgt_length, num_virtual_tokens):
    # apn - Anchor, Positive, Negative
    assert negatives_strategy in ['random', 'code_distance', 'summary_distance', 'momentum_encoder']
    # If "code_distance" is selected, choose negatives by distance between code emb.
    # In this setting, anchor = summary, positive = code, negative = summary whose corr. code is far from positive
    # If "summary_distance" is selected, then
    # anchor = code, positive = summary, negative = code whose corr. summary is far from positive

    # MoCo: https://github.com/facebookresearch/moco/blob/3631be074a0a14ab85c206631729fe035e54b525/moco/builder.py#L6

    return None


def _get_embeddings_loss_fn(loss_type: str):
    if loss_type == 'triplet':
        return torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.CosineSimilarity(), reduction='mean')
    elif loss_type == 'contrastive':
        return None
    elif loss_type == 'sigmoid':
        return None
    else:
        logger.error(f'Unsupported embeddings loss type :{loss_type}')
        exit(-1)


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
                     num_virtual_tokens: int,
                     loss_type: str,
                     model_max_src_length: int = 320,
                     model_max_tgt_length: int = 128,
                     train_batch_size: int = 32,
                     eval_batch_size: int = 16,
                     gradient_accumulation_steps: int = 4,
                     device_type: str = 'cuda:0',
                     model_checkpoint: str = 'Salesforce/codet5p-110m-embedding',
                     ):
    # Ok so I have basically two options:
    #   utilize HF Trainer or write my own training loop (I'll do this if HF fails somewhere)

    device = torch.device(device_type)
    embeddings_model, tokenizer = _setup_embeddings_model(model_checkpoint, num_virtual_tokens, device)

    raw_dataset = DATASET_MAP[language](model_max_src_length)

    apn_dataset = _setup_apn_dataset(raw_dataset, 'random', tokenizer, model_max_src_length, model_max_tgt_length,
                                     num_virtual_tokens)
    # train_loader = DataLoader(seq2seq_dataset['train'], batch_size=train_batch_size, shuffle=True)
    # val_loader = DataLoader(seq2seq_dataset['val'], batch_size=eval_batch_size, shuffle=False)
    # test_loader = DataLoader(seq2seq_dataset['test'], batch_size=eval_batch_size, shuffle=False)

    emb_loss = _get_embeddings_loss_fn(loss_type)

    class EmbeddingsTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            positive = inputs.pop("positive")
            negative = inputs.pop("negative")

            anchor_output = model(**inputs)

            anchor_emb = anchor_output.get("logits")[:, 0, :]
            positive_emb = model(positive).get("logits")[:, 0, :]
            negative_emb = model(negative).get("logits")[:, 0, :]

            loss = emb_loss(anchor_emb, positive_emb, negative_emb)
            return (loss, anchor_output) if return_outputs else loss


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    torch.manual_seed(0)
    random.seed(0)

    from src.datasets.make_datasets import create_python_dataset, create_java_dataset

    DATASET_MAP = {'Python': create_python_dataset, 'Java': create_java_dataset, 'C#': None, 'SQL': None}

    # _setup_embeddings_model('Salesforce/codet5p-110m-embedding', 20, torch.device('cuda:0'))
    Fire(
        {
            'seq2seq': train_seq2seq,
            'embeddings': train_embeddings,
        }
    )
