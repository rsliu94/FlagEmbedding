# export WANDB_MODE=disabled
# 科学上网
source /etc/network_turbo

train_data="\
    ../data/hn_mine_data_zero_round/finetune_data_validation_minedHN_test.jsonl \
"
# set num_gpus to 2 for testing
num_gpus=2
# set large epochs and small batch size for testing
num_train_epochs=1
per_device_train_batch_size=2
per_device_eval_batch_size=16
save_merged_lora_model=False
max_steps=0
learning_rate=1e-4
output_dir="../model_output/icl_ft_debug"

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path BAAI/bge-en-icl \
    --cache_dir $HF_HUB_CACHE \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
    --additional_special_tokens '<instruct>' '<query>' '<response>' \
    --save_merged_lora_model $save_merged_lora_model \
"

data_args="\
    --train_data $train_data \
    --eval_corpus_path ../data/embedder_eval_data/corpus.jsonl \
    --eval_queries_path ../data/embedder_eval_data/queries_val2_v1.jsonl \
    --eval_examples_path ../data/embedder_eval_data/examples_v1.json \
    --cache_path ~/.cache \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 128 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval 'Given a math question and a misconcepted incorrect answer to it, retrieve the most accurate misconception that leads to the incorrect answer.' \
    --query_instruction_format '<instruct>{}\n<query>{}' \
    --knowledge_distillation False \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
    --example_query_max_len 384 \
    --example_passage_max_len 128 \
    --retrieval_use_examples True \
    --icl_suffix_str '\n<response>' \
"

training_args="\
    --save_lora_every_epoch True \
    --max_steps $max_steps \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --learning_rate $learning_rate \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --dataloader_drop_last False \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ../scripts/icl_ds_stage1.json \
    --logging_steps 1 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method last_token \
    --normalize_embeddings True \
    --kd_loss_type kl_div \
"

cmd="torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.embedder.decoder_only.icl \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd