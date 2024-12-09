# Finetune reranker

emb: raw, iter0, iter1
ranker: iter1, iter1

iter0:  [train file name: finetune_data_hn_from_emb_raw.jsonl]
        hn_from_emb_raw -> emb_iter0 -> results [only recall] 
        
iter1:  [file name: finetune_data_hn_from_emb_iter0.jsonl]
        hn_from_emb_iter0 -> emb_iter1; [save intermediate retrieved results]

        [file name: finetune_data_hn_from_emb_iter1.jsonl]
        hn_from_emb_iter1 -> ranker_iter1; 
        [rerank intermediate retrieved results with ranker_iter1, which is the final results of iter1]
        
iter2:  [TBC]

# Iteration 1

## Mine 
* range_for_sampling: [2, 100]
* negative_number: 15
Train data:
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/icl_finetune_iter1_hn/merged_model_lora_epoch_2 \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_from_emb_iter1.jsonl \
--candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
--range_for_sampling 2-100 \
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
Test data:
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/icl_finetune_iter1_hn/merged_model_lora_epoch_2 \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_test_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_from_emb_iter1_test.jsonl \
--candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
--range_for_sampling 2-100 \
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
## Eval raw model
```bash
python eval_llm_reranker.py \
--retrieval_results_path ../model_output/icl_finetune_iter1_hn/retrieval_results.jsonl \
--model_path BAAI/bge-reranker-v2-gemma \
--batch_size 8 \
--device cuda:0
```
==Rerank==
map@25_score: 0.2158348471792536
recall@25_score: 0.6990740740740741
==Recall==
map@25_score: 0.4970577497482009
recall@25_score: 0.8819444444444444

compare hf eval [20min 12G]
```bash
python eval_llm_reranker_hf.py \
--retrieval_results_path ../model_output/icl_finetune_iter1_hn/retrieval_results.jsonl \
--model_path BAAI/bge-reranker-v2-gemma \
--batch_size 8 \
--device cuda:0
```
==Rerank==
map@25_score: 0.22296536644087206
recall@25_score: 0.7175925925925926
==Recall==
map@25_score: 0.4970577497482009
recall@25_score: 0.8819444444444444

## Finetune

train:
ngpu=1, qlora, bs=2, accum=2, lr=1e-4, epoch=1, 30min, from=bge-reranker-v2-gemma

eval: [30min][13G]
```bash
python eval_llm_reranker.py \
--retrieval_results_path ../model_output/icl_finetune_iter1_hn/retrieval_results.jsonl \
--model_path ../model_output/reranker_finetune_iter1/merged_model \
--batch_size 8 \
--device cuda:0
```
==Rerank==
map@25_score: 0.42359769956180326
recall@25_score: 0.8715277777777778
==Recall==
map@25_score: 0.4970577497482009
recall@25_score: 0.8819444444444444

train:
ngpu=4, qlora, bs=2, accum=2, lr=1e-4, epoch=5, from=bge-reranker-v2-gemma / 40min [OOM ???]
ep1: eval_loss=1.39 / rerank map=0.4509 / recall map=0.5499
ep2: eval_loss=2.08 / rerank map=0.4530 / recall map=0.5499 [train loss ~ 0.1]
ep3: eval_loss=2.69 / rerank map=0.4718 / recall map=0.5499 [train loss ~ 0.04]
ep4: eval_loss=3.98 / rerank map=0.5090 / recall map=0.5499 [train loss ~ 0.01]
ep5: eval_loss=4.68 / rerank map=0.4681 / recall map=0.5499 [train loss ~ 0.001]

<!-- UserWarning: Model with `tie_word_embeddings=True` and the tied_target_modules=['lm_head'] are part of the adapter. This can lead to complications. You can opt to merge the adapter after cloning the weights (to untie the embeddings). You can untie the embeddings by loading the model with `tie_word_embeddings=False`. For example:
```python
from transformers import AutoModelForCausalLM

# Load original tied model
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", tie_word_embeddings=False)

# Set the randomly initialized lm_head to the previously tied embeddings
model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

# Save the untied model
untied_model_dir = "dir/for/untied/model"
model.save_pretrained(untied_model_dir)
model.config.save_pretrained(untied_model_dir)

# Now use the original model but in untied format
model = AutoModelForCausalLM.from_pretrained(untied_model_dir)
```

  warnings.warn( -->

train:
ngpu=1, qlora=False, bs=2, accum=2, lr=2e-4, epoch=1, from=bge-reranker-v2-gemma, lora_module: no lm_head
ep1: eval_loss=1.58 / rerank map=0.3401 / recall map=0.5499

train:
ngpu=4, qlora, deepspeed=1, bs=1, accum=8, group=16, lora_dropout=0.05, lr=2e-4, ls=0.0, epoch=1, from=qwen2.5-14b-in, ../model_output/reranker_finetune_iter1_test_qwen_ep1
1hr25min for 1 epoch
```bash
sh reranker_finetune.sh --epochs 1 --batch_size 1 --num_gpus 4 --gpu_ids "0,1,2,3"
```
ep1: eval_loss=0.81 / rerank map=0.3401 / recall map=0.5499
==Rerank==
map@25_score: 0.5729157619785994
recall@25_score: 0.9302325581395349
==Recall==
map@25_score: 0.5499526902711938
recall@25_score: 0.9186046511627907

Eval with script:
```bash
python eval_llm_reranker.py \
--retrieval_results_path ../model_output/icl_finetune_iter1_hn/retrieval_results.jsonl \
--model_path Qwen/Qwen2.5-14B-Instruct \
--lora_path ../model_output/reranker_finetune_iter1_test_qwen_ep1/lora_epoch_0 \
--batch_size 1 \
--device cuda:4
```
```bash
python eval_llm_reranker.py \
--retrieval_results_path ../model_output/icl_finetune_iter1_hn/retrieval_results.jsonl \
--model_path Qwen/Qwen2.5-14B-Instruct \
--batch_size 1 \
--device cuda:0
```

train: ds-2 + gradient_accumulation_steps=4 + 5 epochs
