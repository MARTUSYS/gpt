python3 /home/jup1/GPT-3_news/generate_transformers.py \
    --model_name_or_path=/home/jup1/GPT-3_news/checkpoints/m_line_512_led_all \
    --path_to_prompt=/home/jup1/GPT-3_news/test/1.txt \
    --path_to_save_sample=/home/jup1/GPT-3_news/test/sample.txt \
    --length 64 \
    --temperature 0.9 \
    --k 50 \
    --p 0.95 \
    --pruning \
    --num_return_sequences 3
