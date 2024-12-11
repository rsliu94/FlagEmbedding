
# sh icl_finetune.sh 2>&1 | tee ./logs/icl_hn_finetune_round2_$(date +%Y%m%d_%H%M%S).log
# sh icl_finetune.sh --epochs 1 --batch_size 8 --num_gpus 2 --gpu_ids "0,1" 2>&1 | tee ./logs/icl_finetune_iter0_hn_$(date +%Y%m%d_%H%M%S).log
# export WANDB_MODE=disabled
# 科学上网
source /etc/network_turbo

# 打印初始参数
echo "输入参数: $@"

# 添加参数解析
while [[ $# -gt 0 ]]; do
  key="$1"
  echo "处理参数: $key"
  
  case $key in
    --epochs)
      num_train_epochs="$2"
      echo "设置 epochs = $num_train_epochs"
      shift 2
      ;;
    --batch_size)
      per_device_train_batch_size="$2"
      echo "设置 batch_size = $per_device_train_batch_size"
      shift 2
      ;;
    --train_data)
      train_data="$2"
      echo "设置 train_data = $train_data"
      shift 2
      ;;
    --eval_data)
      eval_data="$2"
      echo "设置 eval_data = $eval_data"
      shift 2
      ;;
    --num_gpus)
      num_gpus="$2"
      echo "设置 num_gpus = $num_gpus"
      shift 2
      ;;
    --gpu_ids)
      export CUDA_VISIBLE_DEVICES="$2"
      echo "设置 CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
      shift 2
      ;;
    --output_dir)
      output_dir="$2"
      echo "设置 output_dir = $output_dir"
      shift 2
      ;;
    --model_name_or_path)
      model_name_or_path="$2"
      echo "设置 model_name_or_path = $model_name_or_path"
      shift 2
      ;;
    --gradient_accumulation_steps)
      gradient_accumulation_steps="$2"
      echo "设置 gradient_accumulation_steps = $gradient_accumulation_steps"
      shift 2
      ;;
    --learning_rate)
      learning_rate="$2"
      echo "设置 learning_rate = $learning_rate"
      shift 2
      ;;
    *)
      echo "错误: 未知参数 '$key'"
      echo "可用参数: --epochs, --batch_size, --num_gpus, --gpu_ids, --train_data, --eval_data, --output_dir, --model_name_or_path, --gradient_accumulation_steps, --learning_rate"
      exit 1
      ;;
  esac
done

# 打印最终设置的值
echo "最终配置:"
echo "num_train_epochs = $num_train_epochs"
echo "per_device_train_batch_size = $per_device_train_batch_size"
echo "num_gpus = $num_gpus"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "model_name_or_path = $model_name_or_path"
echo "gradient_accumulation_steps = $gradient_accumulation_steps"
echo "learning_rate = $learning_rate"
echo "train_data = $train_data"
echo "eval_data = $eval_data"
echo "output_dir = $output_dir"

# 设置默认值
: ${num_train_epochs:=3}
: ${per_device_train_batch_size:=8}
: ${num_gpus:=2}
: ${train_data:="../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl"}
: ${eval_data:="../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl"}
: ${output_dir:="../model_output/icl_finetune_iter0_hn"}
: ${model_name_or_path:="BAAI/bge-en-icl"}
: ${gradient_accumulation_steps:=1}
: ${learning_rate:=2e-4}

eval_corpus_path="../data/embedder_train_eval_data/cross_validation/corpus.jsonl"
eval_queries_path="../data/embedder_train_eval_data/cross_validation/test_queries.jsonl"
eval_examples_path="../data/embedder_train_eval_data/cross_validation/examples.json"

# task_description="Given a multiple choice math question and a student's wrong answer to it, retrieve the math misconception behind the wrong answer."
# set large epochs and small batch size for testing

retrieval_use_examples=True
query_max_len=1024
deepspeed_config_path="./ds_stage1.json"
save_merged_lora_model=True
save_steps=500

lora_rank=32
lora_alpha=64
lora_dropout=0.1
use_qlora=False


if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path $model_name_or_path \
    --cache_dir $HF_HUB_CACHE \
    --use_lora True \
    --lora_rank $lora_rank \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
    --save_merged_lora_model $save_merged_lora_model \
    --use_qlora $use_qlora \
"

training_args="\
    --save_lora_every_epoch True \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --learning_rate $learning_rate \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed $deepspeed_config_path \
    --logging_steps 1 \
    --save_steps $save_steps \
    --negatives_cross_device \
    --save_total_limit 2 \
    --temperature 0.02 \
    --sentence_pooling_method last_token \
    --normalize_embeddings True \
    --kd_loss_type kl_div \
"

data_args="\
    --train_data $train_data \
    --eval_data $eval_data \
    --eval_corpus_path $eval_corpus_path \
    --eval_queries_path $eval_queries_path \
    --eval_examples_path $eval_examples_path \
    --cache_path ~/.cache \
    --train_group_size 8 \
    --query_max_len $query_max_len \
    --passage_max_len 128 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval \"Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer.\" \
    --query_instruction_format '<instruct>{}\n<query>{}' \
    --knowledge_distillation False \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
    --example_query_max_len 384 \
    --example_passage_max_len 128 \
    --retrieval_use_examples $retrieval_use_examples \
    --icl_suffix_str '\n<response>' \
"

random_port=$(shuf -i 29500-65535 -n 1)
cmd="torchrun --nproc_per_node $num_gpus --master_port $random_port \
    -m FlagEmbedding.finetune.embedder.decoder_only.icl \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd