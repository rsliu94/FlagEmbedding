# Finetune Embedder with in context learning
train: try qwen 7b, ds-2
```bash
sh icl_finetune.sh \
--epochs 1 \
--batch_size 16 \
--gradient_accumulation_steps 4 \
--num_gpus 1 \
--gpu_ids "0" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl \
--output_dir ../model_output/embedder_finetune_qwen7b_ep1_ds2 \
--model_name_or_path Qwen/Qwen2.5-7B-Instruct
```
单卡 L20 1epoch bs1*ga1 -> 1s 1.5it / 3504 iter -> 3506/1.5 = 38min / 8G 显存
单卡 L20 1epoch bs4*ga4 -> 7.3s/it / 219 iter -> 7.3 * 219 = 1600s = 27min / 9.9G 显存
单卡 L20 1epoch bs16*ga4 -> 35min

Metrics性能参考：
lora r=32,a=64,lr=1e-4, bs=8*4 on bge-en-icl
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | 1.2708 | 0.4683 | 0.8692/0.9247/0.9502/0.9629 | ? |

lora r=32,a=64,lr=1e-4, bs=8*2, on mistral-7b [note: add 3 special tokens, emb changed]
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | 0.9872 | 0.4255 | 0.8506/0.9097/0.9421/0.9517 | ? |

lora r=32,a=64,lr=1e-4, bs=16*4[单卡] on qwen-7b-instruct
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | 1.4548 | 0.3037 | 0.7627/0.8472/0.8773/0.9016 | ? |

train: try qwen 14b, ds-2
```bash
sh icl_finetune.sh \
--epochs 1 \
--batch_size 8 \
--gradient_accumulation_steps 2 \
--num_gpus 2 \
--gpu_ids "0,1" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl \
--output_dir ../model_output/embedder_icl_finetune_qwen14b_ep1_ds2 \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct
```
2卡 L20 1epoch bs8*ga2*2 -> 25min / 35G 显存
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | 1.1267 | 0.3622 | 0.8263/0.8888/0.9085/0.9270 | ? |

train: try qwen 14b, ds-2, 3epochs, 4gpus [32min/3epochs]
```bash
sh icl_finetune.sh \
--epochs 3 \
--batch_size 8 \
--gradient_accumulation_steps 2 \
--num_gpus 4 \
--gpu_ids "0,1,2,3" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl \
--output_dir ../model_output/embedder_icl_finetune_qwen14b_ep3_ds2 \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct
```
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | 1.5497 | 0.3682 | 0.8194/0.8842/0.9236/0.9317 | ? |
| 2     | 1.2083 | 0.4582 | 0.8773/0.9270/0.9467/0.9618 | ? |
| 3     | 1.1417 | 0.4659 | 0.8854/0.9375/0.9548/0.9606 | ? |

When Finetune with qwen-14b, should not add special tokens.
单卡 2 epochs
```bash
sh icl_finetune.sh \
--epochs 2 \
--batch_size 8 \
--gradient_accumulation_steps 2 \
--num_gpus 1 \
--gpu_ids "0" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl \
--output_dir ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct
```
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | 0.6799 | 0.4444 | 0.8912/0.9386/0.9537/0.9606 | ? |
| 2     | 0.6446 | 0.4714 | 0.8969/0.9259/0.9513/0.9594 | ? |

2 epochs / 2 gpus / lr 1e-4 ga=2
```bash
sh icl_finetune.sh \
--epochs 2 \
--batch_size 8 \
--gradient_accumulation_steps 2 \
--num_gpus 2 \
--gpu_ids "0,1" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl \
--output_dir ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter0 \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct
```
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | 0.9577 | 0.4444 | 0.8680/0.9131/0.9340/0.9502 | ? |
| 2     | 0.8767 | 0.4676 | 0.8865/0.9236/0.9409/0.9560 | ? |

2 epochs / 2 gpus / lr 1e-4 ga=1 [Best Recipe][1hr]
```bash
sh icl_finetune.sh \
--epochs 2 \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--num_gpus 2 \
--learning_rate 1e-4 \
--gpu_ids "0,1" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl \
--output_dir ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter0 \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct
```
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | 0.8871 | 0.4547 | 0.8784/0.9247/0.9375/0.9560 | ? |
| 2     | 0.8026 | 0.4853 | 0.9016/0.9363/0.9490/0.9606 | CHECK LB |

2 epochs / 4 gpus / lr 2e-4 ga=1 [25min][???? insanely good?]
```bash
sh icl_finetune.sh \
--epochs 2 \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--num_gpus 4 \
--learning_rate 2e-4 \
--gpu_ids "0,1,2,3" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl \
--output_dir ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter0_v3 \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct
```
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | 1.194 | 0.4665 | 0.8773/0.9280/0.9502/0.9664 | ? |
| 2     | 1.0966 | 0.5031 | 0.8958/0.9363/0.9583/0.9722 | CHECK LB |


2 epochs / 2 gpus / lr 1e-4 ga=1 + wd + beta2
```bash
sh icl_finetune.sh \
--epochs 2 \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--num_gpus 2 \
--gpu_ids "2,3" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl \
--output_dir ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter0_v2 \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct
```
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | 0.9247 | 0.4437 | 0.8726/0.9212/0.9479/0.9606 | ? |
| 2     | 0.8043 | 0.4850 | 0.8935/0.9293/0.9490/0.9618 | ? |


check eval score metrics:
```bash
python eval_llm_embedder.py \
--use_examples_in_query True \
--model_path Qwen/Qwen2.5-14B-Instruct \
--lora_path ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb/lora_epoch_2/adapter_model.bin \
--query_max_len 512 \
--use_examples_in_query True \
--device cuda:0 \
--save_retrieval_results True \
--k 25 \
--batch_size 8 \
--retrieval_results_path ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb/retrieval_results_top25.jsonl
```
12G 显存
map@25_score: 0.002092476370458397
recall@25_score: 0.011574074074074073
map@50_score: 0.0024807797553062032
recall@50_score: 0.027777777777777776
map@75_score: 0.0026593137010856175
recall@75_score: 0.03935185185185185
map@100_score: 0.0028490933815474026
recall@100_score: 0.056712962962962965

try raw model without lora
```bash
python eval_llm_embedder.py \
--use_examples_in_query True \
--model_path Qwen/Qwen2.5-14B-Instruct \
--query_max_len 512 \
--use_examples_in_query True \
--device cuda:0 \
--save_retrieval_results True \
--k 25 \
--batch_size 8 \
--retrieval_results_path ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb/retrieval_results_top25.jsonl
```
map@25_score: 0.0018818377541839277
recall@25_score: 0.009259259259259259
map@50_score: 0.002337084500092505
recall@50_score: 0.02662037037037037
map@75_score: 0.002507788858639991
recall@75_score: 0.037037037037037035
map@100_score: 0.002726782859471331
recall@100_score: 0.056712962962962965
```bash
python eval_llm_embedder.py \
--use_examples_in_query True \
--model_path ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb/merged_model \
--query_max_len 512 \
--use_examples_in_query True \
--device cuda:0 \
--save_retrieval_results True \
--k 25 \
--batch_size 8 \
--retrieval_results_path ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb/retrieval_results_top25.jsonl
```
map@25_score: 0.45625847627561184
recall@25_score: 0.8842592592592593
map@50_score: 0.4577198025180625
recall@50_score: 0.9328703703703703
map@75_score: 0.4579904776925519
recall@75_score: 0.9490740740740741
map@100_score: 0.45814271042662846
recall@100_score: 0.9618055555555556


# Revert to my old lora saver and re try
单卡 2 epochs
```bash
sh icl_finetune.sh \
--epochs 2 \
--batch_size 8 \
--gradient_accumulation_steps 2 \
--num_gpus 1 \
--gpu_ids "0" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl \
--output_dir ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb_old_saver \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct
```
try eval again  12G 显存 / 
```bash
python eval_llm_embedder.py \
--use_examples_in_query True \
--model_path Qwen/Qwen2.5-14B-Instruct \
--lora_path ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb_old_saver/lora_epoch_2 \
--query_max_len 512 \
--use_examples_in_query True \
--device cuda:0 \
--save_retrieval_results True \
--k 25 \
--batch_size 8 \
--retrieval_results_path ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb_old_saver/retrieval_results_top25.jsonl
```
map@25_score: 0.03384432133303275
recall@25_score: 0.1284722222222222
map@50_score: 0.035964684792512976
recall@50_score: 0.2037037037037037
map@75_score: 0.036622461030017414
recall@75_score: 0.24305555555555555
map@100_score: 0.03715902455028206 
????
```bash
python eval_llm_embedder.py \
--use_examples_in_query True \
--model_path ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb_old_saver/merged_model \
--query_max_len 512 \
--use_examples_in_query True \
--device cuda:0 \
--save_retrieval_results True \
--k 25 \
--batch_size 8 \
--retrieval_results_path ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb_old_saver/retrieval_results_top25.jsonl
```
map@25_score: 0.45625847627561184
recall@25_score: 0.8842592592592593
map@50_score: 0.4577198025180625
recall@50_score: 0.9328703703703703
map@75_score: 0.4579904776925519
recall@75_score: 0.9490740740740741
map@100_score: 0.45814271042662846
recall@100_score: 0.9618055555555556
```bash
python eval_llm_embedder.py \
--use_examples_in_query True \
--model_path Qwen/Qwen2.5-14B-Instruct \
--lora_path ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb_old_saver/ \
--query_max_len 512 \
--use_examples_in_query True \
--device cuda:0 \
--save_retrieval_results True \
--k 25 \
--batch_size 8 \
--retrieval_results_path ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb_old_saver/retrieval_results_top25.jsonl
```
map@25_score: 0.03384432133303275
recall@25_score: 0.1284722222222222
map@50_score: 0.035964684792512976
recall@50_score: 0.2037037037037037
map@75_score: 0.036622461030017414
recall@75_score: 0.24305555555555555
map@100_score: 0.03715902455028206
recall@100_score: 0.29050925925925924

去掉merge_and_unload, 13.8G 显存
```bash
python eval_llm_embedder.py \
--use_examples_in_query True \
--model_path Qwen/Qwen2.5-14B-Instruct \
--lora_path ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb_old_saver/lora_epoch_2 \
--query_max_len 512 \
--use_examples_in_query True \
--device cuda:0 \
--save_retrieval_results True \
--k 25 \
--batch_size 8 \
--retrieval_results_path ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb_old_saver/retrieval_results_top25.jsonl
```
map@25_score: 0.45639742799935207
recall@25_score: 0.8854166666666666
map@50_score: 0.4578993747077715
recall@50_score: 0.9351851851851852
map@75_score: 0.458228303805246
recall@75_score: 0.9548611111111112
map@100_score: 0.4582945450617536

Finally!!!!!!!!!!!!!!

多线程 1min for query
```bash
python eval_llm_embedder_multi_device.py \
--use_examples_in_query True \
--model_path Qwen/Qwen2.5-14B-Instruct \
--lora_path ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb_old_saver/lora_epoch_2 \
--query_max_len 512 \
--use_examples_in_query True \
--device cuda:0,cuda:1 \
--save_retrieval_results True \
--k 25 \
--batch_size 8 \
--retrieval_results_path ../model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb_old_saver/retrieval_results_top25.jsonl
```
map@25_score: 0.4560116255302163
recall@25_score: 0.8854166666666666
map@50_score: 0.45751441716339275
recall@50_score: 0.9351851851851852
map@75_score: 0.45784334626086726
recall@75_score: 0.9548611111111112
map@100_score: 0.4579097320126816
recall@100_score: 0.9606481481481481

# HN using qwen14b embedder
projects/model_output/embedder_icl_finetune_qwen14b_ep2_ds2_fix_emb_old_saver/merged_model