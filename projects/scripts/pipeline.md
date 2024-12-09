# Iter-0
# 1. Hard negative mining using bge-en-icl
* range_for_sampling: [2, 200]
* negative_number: 15

Train data:
```bash
python hn_mine.py \
--embedder_name_or_path BAAI/bge-en-icl \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl \
--candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--devices cuda:1 \
--shuffle_data True \
--query_instruction_for_retrieval "Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer." \
--query_instruction_format '<instruct>{}\n<query>{}' \
--add_examples_for_task True \
--batch_size 1024 \
--embedder_query_max_length 1024 \
--embedder_passage_max_length 512
```

For test data:
```bash
python hn_mine.py \
--embedder_name_or_path BAAI/bge-en-icl \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_test_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl \
--candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--devices cuda:0 \
--shuffle_data True \
--query_instruction_for_retrieval "Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer." \
--query_instruction_format '<instruct>{}\n<query>{}' \
--add_examples_for_task True \
--batch_size 1024 \
--embedder_query_max_length 1024 \
--embedder_passage_max_length 512
```

# 2. Finetune Qwen2.5-7B-Instruct with Hard negative mining results [Using ICL]
```bash
sh icl_finetune.sh \
--epochs 1 \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--num_gpus 2 \
--gpu_ids "0,1" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl \
--output_dir ../model_output/icl_finetune_iter0_hn \
--model_name_or_path Qwen/Qwen2.5-7B-Instruct
```

# 3. Retrieval using Qwen2.5-7B-Instruct