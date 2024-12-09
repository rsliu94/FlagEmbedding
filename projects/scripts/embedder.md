# Finetune embedder
train: try qwen 7b, ds-2
```bash
sh embedder_finetune.sh \
--epochs 1 \
--batch_size 16 \
--gradient_accumulation_steps 1 \
--num_gpus 1 \
--gpu_ids "0" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl \
--output_dir ../model_output/embedder_base_finetune_qwen7b_ep1_ds2 \
--model_name_or_path Qwen/Qwen2.5-7B-Instruct
```

单卡 L20 1epoch bs16*ga1 -> 21.3s/it -> 21.3 * 54 = 18min / 12G 显存
lora r=32,a=64,lr=1e-4, bs=16*1[单卡] on qwen-7b-instruct
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | N/A | 0.2778 | 0.7569/0.8391/0.8900/0.9097 | N/A |

train: try qwen 14b, ds-2
```bash
sh embedder_finetune.sh \
--epochs 1 \
--batch_size 8 \
--gradient_accumulation_steps 2 \
--num_gpus 1 \
--gpu_ids "1" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl \
--output_dir ../model_output/embedder_base_finetune_qwen14b_ep1_ds2 \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct
```
