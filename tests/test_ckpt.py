import os

import pytest
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModel, AutoTokenizer

CHECKPOINTS_PATH = 'checkpoints'
MODELS_DIRS = os.listdir(CHECKPOINTS_PATH)
PATHS = []
for _dir in MODELS_DIRS:
    for ckpt_dir in os.listdir(os.path.join(CHECKPOINTS_PATH, _dir)):
        PATHS.append(os.path.join(CHECKPOINTS_PATH, _dir, ckpt_dir))


@pytest.mark.parametrize('tuned_ckpt_path, sample_input', [(path, "Your sample text") for path in PATHS])
def test_model(tuned_ckpt_path, sample_input):
    config = PeftConfig.from_pretrained(tuned_ckpt_path)

    model = AutoModel.from_pretrained(config.base_model_name_or_path, device_map={"": 0}, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, tuned_ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Encode the sample input
    encoded_input = tokenizer.encode_plus(sample_input, return_tensors="pt", padding='max_length', max_length=54, truncation=True)
    input_ids = encoded_input['input_ids'].to(next(model.parameters()).device)

    if 'seq2seq' in str(type(model)).lower():
        output = model(input_ids, decoder_input_ids=input_ids)
    else:
        output = model(input_ids)

    # Get the predicted token ids from the output tensor
    predicted_token_ids = torch.argmax(output.logits, dim=-1)

    # Decode the token ids back to text
    decoded_text = tokenizer.decode(predicted_token_ids[0])

    return decoded_text

print(test_model(PATHS[0], "draw the circle"))