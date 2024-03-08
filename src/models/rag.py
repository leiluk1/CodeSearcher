import os
import sys
import warnings
from typing import List

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from fire import Fire
from langchain.docstore.document import Document as LangchainDocument
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.pydantic_v1 import BaseModel
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from loguru import logger
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, DataCollatorForSeq2Seq, pipeline, AutoModelForCausalLM

PROMPT_IN_CHAT_FORMAT = [
    {
        "role": "system",
        "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """Context:
{context}
---
Now here is the question you need to answer.

Question: {question}""",
    },
]


class CodeT5PlusEmbedder(BaseModel, Embeddings, extra='allow'):
    def __init__(self, tuned_ckpt_path, device_type='cuda:0', pbar=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(device_type)
        config = PeftConfig.from_pretrained(tuned_ckpt_path)

        model = AutoModel.from_pretrained(config.base_model_name_or_path, device_map={"": 0}, trust_remote_code=True)
        self.model = PeftModel.from_pretrained(model, tuned_ckpt_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)

        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=config.base_model_name_or_path)
        self.pbar = pbar

    def _inference(self, texts: List[str]) -> List[List[float]]:
        model_inputs = [self.tokenizer(text, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
                        for text in texts]
        loader = DataLoader(model_inputs, batch_size=32, shuffle=False, collate_fn=self.data_collator)
        embeddings = []
        batches = loader if not self.pbar else tqdm(loader, total=len(loader), desc='Iterating over documents')
        for batch in batches:
            batch['input_ids'] = batch['input_ids'].squeeze(1)
            batch['attention_mask'] = batch['attention_mask'].squeeze(1)
            batch.to(self.device)
            with torch.no_grad():
                embeddings.append(self.model(**batch).cpu())
        embeddings = np.concatenate(embeddings, 0).reshape(-1, embeddings[0].shape[-1])
        return embeddings.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._inference(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._inference([text])[0]


def construct_pipeline(tuned_ckpt_path, language, reader_name, device_type='cuda:0', is_interactive=True):
    embedder = CodeT5PlusEmbedder(tuned_ckpt_path, device_type=device_type, pbar=is_interactive)
    logger.info('Embedder model created')
    raw_dataset = load_dataset("code_search_net", language)['test'].to_pandas()
    dataset = raw_dataset['func_code_string'].to_list()
    code_chunks = [LangchainDocument(sample) for sample in dataset]
    logger.info('Started vector DB construction')
    db = FAISS.from_documents(code_chunks, embedder, distance_strategy=DistanceStrategy.COSINE)

    tokenizer = AutoTokenizer.from_pretrained(reader_name, padding=True, truncation=True,
                                              max_length=512, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(reader_name, trust_remote_code=True).to(torch.device(device_type))

    code_generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=512,
        device=next(model.parameters()).device
    )

    reader_llm = HuggingFacePipeline(
        pipeline=code_generator,
        model_kwargs={"temperature": 0.5, "max_length": 512, "device": "cuda"},
    )
    logger.info('Reader LLM created')

    rag_prompt = tokenizer.apply_chat_template(
        PROMPT_IN_CHAT_FORMAT, tokenize=False, add_generation_prompt=True
    )
    logger.info('RAG pipeline ready')
    if is_interactive:
        return reader_llm, db, rag_prompt
    else:
        return reader_llm, db, rag_prompt, raw_dataset


def rag_inference(reader_llm, vector_db, rag_prompt, query):
    retrieved_docs = vector_db.similarity_search(query=query, k=5)
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

    final_prompt = rag_prompt.format(question=query, context=context)

    answer = reader_llm(final_prompt)
    return answer


def rag_benchmarking(embedder_path,
                     language,
                     reader_name="deepseek-ai/deepseek-coder-1.3b-base",
                     device_type="cuda"):
    reader_llm, vector_db, rag_prompt, dataset = construct_pipeline(embedder_path, reader_name=reader_name,
                                                                    language=language, device_type=device_type,
                                                                    is_interactive=False)
    rouge = evaluate.load("rouge")
    labels = []
    predictions = []
    max_samples = 1000
    for query, gt_code in tqdm(zip(dataset['func_documentation_string'][:max_samples],
                                   dataset['func_code_string'][:max_samples]),
                               total=max_samples,
                               desc='Calculating ROUGE'):
        response = rag_inference(reader_llm, vector_db, rag_prompt, query)
        predictions.append(response)
        labels.append(gt_code)

    result = rouge.compute(predictions=predictions, references=labels, use_stemmer=True)
    logger.info(f'ROUGE of RAG generation = {result}')


def rag_interactive_inference(embedder_path,
                              language,
                              reader_name="deepseek-ai/deepseek-coder-1.3b-base",
                              device_type="cuda"):
    reader_llm, vector_db, rag_prompt = construct_pipeline(embedder_path, reader_name=reader_name,
                                                           language=language, device_type=device_type)

    while True:
        user_query = input("Enter your query. To finish inference, enter <<END>>: ")
        if user_query == '<<END>>':
            break
        rag_answer = rag_inference(reader_llm, vector_db, rag_prompt, user_query)
        print("RAG answer: ")
        print(rag_answer)
        print('\n', '=' * 45)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    warnings.filterwarnings('ignore')

    Fire(
        {
            'interactive': rag_interactive_inference,
            'benchmarking': rag_benchmarking
        }
    )
