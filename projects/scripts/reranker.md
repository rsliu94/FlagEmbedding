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
```bash
python eval_llm_embedder.py \
--use_examples_in_query True \
--model_path BAAI/bge-en-icl \
--lora_path ../model_output/icl_finetune_iter1_hn/lora_epoch_2 \
--query_max_len 512 \
--device cuda:0 \
--save_retrieval_results True \
--k 25 \
--retrieval_results_path ../model_output/icl_finetune_iter1_hn/retrieval_results_top25.jsonl
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

train: try qwen 7b, ds-2
```bash
sh reranker_finetune.sh \
--epochs 1 \
--batch_size 1 \
--gradient_accumulation_steps 1 \
--num_gpus 1 \
--gpu_ids "0" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_from_emb_iter1.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_from_emb_iter1_test.jsonl \
--model_name_or_path Qwen/Qwen2.5-7B-Instruct \
--output_dir ../model_output/reranker_ft_qwen7b_ep1_ds2
```
单卡 L20 1epoch bs1*ga1 -> 1.4s 一个 batch / 3504 iter -> 1.4 * 3504 = 4905.6s = 1hr30min / 14G 显存
单卡 L20 1epoch bs4*ga1 -> 8s 一个 batch / 876 iter -> 8 * 876 = 6992s = 1hr40min / 37G 显存
单卡 L20 1epoch bs4*ga4 ->

train: try qwen 14b, ds-2
```bash
sh reranker_finetune.sh \
--epochs 1 \
--batch_size 2 \
--gradient_accumulation_steps 8 \
--num_gpus 1 \
--gpu_ids "0" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_from_emb_iter1.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_from_emb_iter1_test.jsonl \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct \
--output_dir ../model_output/reranker_ft_qwen14b_ep1_ds2
```
单卡 L20 1epoch bs1*ga1 -> 3s 一个 batch / 3504 iter -> 3 * 3504 = 10512s = 2hr50min / 21G 显存
单卡 L20 1epoch bs2*ga1 -> 6s 一个 batch / 1753 iter -> 6 * 1753 = 10518s = 2hr55min / 30G 显存 
单卡 L20 1epoch bs2*ga8 -> 56s 一个 batch / 219 iter -> 56 * 219 = 12264s = 2hr25min / 32G 显存 / 4卡 50min


train: qwen 7b, 4 gpus
train_group_size=16, qlora=False, deepspeed=2, dropout=0.05, lr=2e-4, ls=0.0, epoch=5, 显存30G, 30min/epoch
```bash
sh reranker_finetune.sh \
--epochs 5 \
--batch_size 2 \
--gradient_accumulation_steps 8 \
--num_gpus 4 \
--gpu_ids "0,1,2,3" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_from_emb_iter1.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_from_emb_iter1_test.jsonl \
--model_name_or_path Qwen/Qwen2.5-7B-Instruct \
--output_dir ../model_output/reranker_ft_qwen7b_5ep_4gpu
```
ep1: eval_loss=0.98
==Rerank==
map@25_score: 0.5061454662908151
recall@25_score: 0.8953488372093024
==Recall==
map@25_score: 0.5499526902711938
recall@25_score: 0.9186046511627907

ep2: eval_loss=2.56
==Rerank==
map@25_score: 0.5218941812101867
recall@25_score: 0.9418604651162791
==Recall==
map@25_score: 0.5499526902711938
recall@25_score: 0.9186046511627907

ep3: eval_loss=3.47
==Rerank==
map@25_score: 0.5406017211402769
recall@25_score: 0.9186046511627907
==Recall==
map@25_score: 0.5499526902711938
recall@25_score: 0.9186046511627907

ep4: eval_loss=5.16
map@25_score: 0.5365388395773953
recall@25_score: 0.9186046511627907
==Recall==
map@25_score: 0.5499526902711938
recall@25_score: 0.9186046511627907

```bash
sh reranker_finetune.sh \
--epochs 2 \
--batch_size 2 \
--gradient_accumulation_steps 8 \
--num_gpus 4 \
--gpu_ids "0,1,2,3" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_from_emb_iter1.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_from_emb_iter1_test.jsonl \
--model_name_or_path Qwen/Qwen2.5-7B-Instruct \
--output_dir ../model_output/reranker_ft_qwen7b_2ep_4gpu
```
On Top-25 results
ep1: eval_loss=2.33
==Rerank==
map@25_score: 0.5061244844722764
recall@25_score: 0.8831018518518519
==Recall==
map@25_score: 0.49682847285225723
recall@25_score: 0.8831018518518519

ep2: eval_loss=2.64
==Rerank==
map@25_score: 0.5414676648680952
recall@25_score: 0.8831018518518519
==Recall==
map@25_score: 0.49682847285225723
recall@25_score: 0.8831018518518519

train: qwen 14b, 4 gpus
```bash
sh reranker_finetune.sh \
--epochs 4 \
--batch_size 2 \
--gradient_accumulation_steps 8 \
--num_gpus 4 \
--gpu_ids "0,1,2,3" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_from_emb_iter1.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_from_emb_iter1_test.jsonl \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct \
--output_dir ../model_output/reranker_ft_qwen14b_ep4_4gpu \
2>&1 | tee ./logs/reranker_finetune_qwen14b_ep4_$(date +%Y%m%d_%H%M%S).log ; /usr/bin/shutdown
```
On Top-25 results
ep1: eval_loss=0.7737
==Rerank==
map@25_score: 0.5760428660883113
recall@25_score: 0.8831018518518519
==Recall==
map@25_score: 0.49682847285225723
recall@25_score: 0.8831018518518519

ep2: eval_loss=0.9893
==Rerank==
map@25_score: 0.5919033183796847
recall@25_score: 0.8831018518518519
==Recall==
map@25_score: 0.49682847285225723
recall@25_score: 0.8831018518518519

ep3: eval_loss=1.38
==Rerank==
map@25_score: 0.6004591938191238
recall@25_score: 0.8831018518518519
==Recall==
map@25_score: 0.49682847285225723
recall@25_score: 0.8831018518518519

ep4: eval_loss=1.60
==Rerank==
map@25_score: 0.603093928403409
recall@25_score: 0.8831018518518519
==Recall==
map@25_score: 0.49682847285225723
recall@25_score: 0.8831018518518519
