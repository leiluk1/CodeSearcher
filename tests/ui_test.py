from transformers import AutoTokenizer, AutoModel
import torch
from peft import PeftModel, PeftConfig


def get_decoded_text_from_model(checkpoint_path, input_text):
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

    # Get the model's output
    with torch.no_grad():
        output = model(input_ids, decoder_input_ids=input_ids)

    # Get the predicted token ids from the output tensor
    predicted_token_ids = torch.argmax(output.logits, dim=-1)

    # Decode the token ids back to text
    decoded_text = tokenizer.decode(predicted_token_ids[0])

    return decoded_text

# checkpoint_path = "checkpoints/codet5p-220m-seq2seq/prefix-python/"
# input_text = "print hello world"
# decoded_text = get_decoded_text_from_model(checkpoint_path, input_text)
# print(decoded_text)