python src/models/train.py embeddings lora \
    --output_dir="output" \
    --epochs=1 \
    --language="C++" \
    --lora_r=8 \
    --device_type="cuda:0" \
    --train_batch_size=16 \
    --eval_batch_size=16 \
    --model_max_src_length=64 \
    --model_max_tgt_length=64