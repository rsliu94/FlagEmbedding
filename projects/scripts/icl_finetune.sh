# export WANDB_MODE=disabled
# 科学上网
source /etc/network_turbo

train_data="\
    ../data/hn_mine_data_zero_round/finetune_data_validation_minedHN.jsonl \
"

# set large epochs and small batch size for testing
num_train_epochs=1
per_device_train_batch_size=16

# set num_gpus to 2 for testing
num_gpus=1

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
    --save_merged_lora_model True \
"

data_args="\
    --train_data $train_data \
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
    --output_dir ../model_output/test_decoder_only_base_bge-en-icl_sd \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --dataloader_drop_last False \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ./icl_ds_stage1.json \
    --logging_steps 5 \
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