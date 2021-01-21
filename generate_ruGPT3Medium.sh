python generate_transformers.py \
    --model_name_or_path=sberbank-ai/rugpt3medium_based_on_gpt2 \
    --path_to_prompt=sample/1.txt \
    --path_to_save_sample=sample/sample.txt \
    --length=128 \
    --temperature=0.9
    --k=50 \
    --p=0.95 \
    --num_return_sequences 3
