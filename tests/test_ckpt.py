import os

import pytest
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModel

CHECKPOINTS_PATH = 'checkpoints'
MODELS_DIRS = os.listdir(CHECKPOINTS_PATH)
PATHS = []
for _dir in MODELS_DIRS:
    for ckpt_dir in os.listdir(os.path.join(CHECKPOINTS_PATH, _dir)):
        PATHS.append(os.path.join(CHECKPOINTS_PATH, _dir, ckpt_dir))


@pytest.mark.parametrize('tuned_ckpt_path', PATHS)
def test_model(tuned_ckpt_path):
    config = PeftConfig.from_pretrained(tuned_ckpt_path)

    model = AutoModel.from_pretrained(config.base_model_name_or_path, device_map={"": 0}, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, tuned_ckpt_path)
    sample_input = torch.zeros(size=(4, 54), device=next(model.parameters()).device, dtype=torch.int32)
    if 'seq2seq' in str(type(model)).lower():
        model(sample_input, decoder_input_ids=sample_input)
    else:
        model(sample_input)
