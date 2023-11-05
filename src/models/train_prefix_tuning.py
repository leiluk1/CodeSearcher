import os

import evaluate
import numpy as np
import torch
from fire import Fire
from loguru import logger
from peft import PrefixTuningConfig, TaskType, get_peft_model
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, AutoModelForSeq2SeqLM, \
    AutoTokenizer

from src.datasets.make_datasets import create_python_dataset

DATASET_MAP = {'Python': create_python_dataset, 'Java': None, 'C#': None, 'SQL': None}


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
                                        num_proc=-1,
                                        )
    return tokenized_dataset


def train_seq2seq(model_checkpoint,
                  output_dir: str,
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
        weight_decay=0.001,
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


if __name__ == '__main__':
    Fire(train_seq2seq)
