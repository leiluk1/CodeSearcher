from transformers import AutoTokenizer, AutoModel, DataCollatorForSeq2Seq
import torch
from peft import PeftModel, PeftConfig
import numpy as np
from annoy import AnnoyIndex


from src.fixtures import DATASET_MAP

from src.models.train import _setup_seq2seq_dataset


def find_nearest(embedding, index, id_map, k=1):
    nearest_ids = index.get_nns_by_vector(embedding, k)
    nearest_codes = [id_map[i] for i in nearest_ids]
    return nearest_codes

def get_decoded_text_from_model(checkpoint_path, input_text, language='SQL'):
    device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device_type)
    config = PeftConfig.from_pretrained(checkpoint_path)
    # Load the model and tokenizer
    model = AutoModel.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.to(device)  # Move the model to the desired device
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    # Encode the input text
    encoded_input = tokenizer.encode_plus(input_text, return_tensors="pt", padding='max_length', max_length=54, truncation=True).to(device)
    input_ids = encoded_input['input_ids'].to(device)  # Move the input tensor to the same device as the model

    is_encoder_only = 'seq2seq' not in str(type(model)).lower()
    if is_encoder_only:
        output = model(**encoded_input).cpu()
    else:
        output = model.encoder(encoded_input['input_ids']).last_hidden_state[:, 0, :].cpu()

    output = output.view(-1)  # Reshape the output tensor to a 1D tensor    

    raw_dataset = DATASET_MAP[language](max_length=128)

    tokenized_dataset = _setup_seq2seq_dataset(raw_dataset, tokenizer, 128, 128, 0)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=config.base_model_name_or_path)

    test_loader = torch.utils.data.DataLoader(tokenized_dataset['test'], batch_size=16, shuffle=False, collate_fn=data_collator)

    test_text_embeddings = []
    test_code_embeddings = []

    for batch in test_loader:
        batch.to(device)
        text_batch = batch.pop("labels")

        with torch.no_grad():
            if is_encoder_only:
                test_code_embeddings.append(model(**batch).cpu())
                test_text_embeddings.append(model(text_batch, attention_mask=(text_batch != 0).int()).cpu())
            else:
                test_code_embeddings.append(model.encoder(batch['input_ids']).last_hidden_state[:, 0, :].cpu())
                test_text_embeddings.append(model.encoder(text_batch).last_hidden_state[:, 0, :].cpu())
        if len(test_code_embeddings) == float('inf'):
            break

    test_text_embeddings = np.concatenate(test_text_embeddings, 0)
    test_code_embeddings = np.concatenate(test_code_embeddings, 0)


    # Assume embeddings is a list of your code embeddings and codes is a list of the corresponding codes
    embeddings = test_code_embeddings

    codes = raw_dataset['test']['code_tokens']

    # Build the Annoy index
    index = AnnoyIndex(embeddings[1], 'angular')  # Length of item vector that will be indexed
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)
    index.build(10)  # 10 trees

    # Build the id map
    id_map = {i: code for i, code in enumerate(codes)}

    # Find the nearest code
    nearest_codes = find_nearest(output, index, id_map, k=1)
    
    return nearest_codes

# checkpoint_path = "checkpoints/codet5p-220m-seq2seq/prefix-sql/"
# input_text = "print hello world"
# decoded_text = get_decoded_text_from_model(checkpoint_path, input_text)

# print(decoded_text)