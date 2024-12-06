# Hard Negatives mining

## [round 0] Mine with pretrained embedder

### Mine
* range_for_sampling: [2, 200]
* negative_number: 15

```bash
python hn_mine.py \
--embedder_name_or_path BAAI/bge-en-icl \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_mined_round0.jsonl \
--candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--devices cuda:1 \
--shuffle_data True \
--query_instruction_for_retrieval "Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer." \
--query_instruction_format '<instruct>{}\n<query>{}' \
--add_examples_for_task False \
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
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_mined_round0_test.jsonl \
--candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--devices cuda:0 \
--shuffle_data True \
--query_instruction_for_retrieval "Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer." \
--query_instruction_format '<instruct>{}\n<query>{}' \
--add_examples_for_task False \
--batch_size 1024 \
--embedder_query_max_length 1024 \
--embedder_passage_max_length 512
```
### Finetune
