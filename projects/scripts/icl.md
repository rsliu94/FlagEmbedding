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