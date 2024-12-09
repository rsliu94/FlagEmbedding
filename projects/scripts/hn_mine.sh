# python hn_mine.py \
# --embedder_name_or_path ../model_output/icl_finetune_round1/merged_model_lora_epoch_1 \
# --embedder_model_class decoder-only-icl \
# --pooling_method last_token \
# --input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_input.jsonl \
# --output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_mined_round1.jsonl \
# --candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
# --range_for_sampling 2-200 \
# --negative_number 15 \
# --devices cuda:0 \
# --shuffle_data True \
# --query_instruction_for_retrieval "Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer." \
# --query_instruction_format '<instruct>{}\n<query>{}' \
# --add_examples_for_task True \
# --batch_size 1024 \
# --embedder_query_max_length 1024 \
# --embedder_passage_max_length 512
# sh hn_mine.sh 2>&1 | tee ./logs/hn_mine_$(date +%Y%m%d_%H%M%S).log
# sh hn_mine.sh --batch_size 1024 --gpu_ids "0" 2>&1 | tee ./logs/hn_mine_$(date +%Y%m%d_%H%M%S).log

source /etc/network_turbo

# 打印初始参数
echo "输入参数: $@"

# 添加参数解析
while [[ $# -gt 0 ]]; do
  key="$1"
  echo "处理参数: $key"
  
  case $key in
    --batch_size)
      batch_size="$2"
      echo "设置 batch_size = $batch_size"
      shift 2
      ;;
    --gpu_ids)
      export CUDA_VISIBLE_DEVICES="$2"
      echo "设置 CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
      shift 2
      ;;
    --input_file)
      input_file="$2"
      echo "设置 input_file = $input_file"
      shift 2
      ;;
    --output_file)
      output_file="$2"
      echo "设置 output_file = $output_file"
      shift 2
      ;;
    --embedder_name_or_path)
      embedder_name_or_path="$2"
      echo "设置 embedder_name_or_path = $embedder_name_or_path"
      shift 2
      ;;
    --candidate_pool)
      candidate_pool="$2"
      echo "设置 candidate_pool = $candidate_pool"
      shift 2
      ;;
    *)
      echo "错误: 未知参数 '$key'"
      echo "可用参数: --batch_size, --gpu_ids, --input_file, --output_file, --embedder_name_or_path, --candidate_pool"
      exit 1
      ;;
  esac
done

# 打印最终设置的值
echo "最终配置:"
echo "batch_size = $batch_size"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "input_file = $input_file"
echo "output_file = $output_file"
echo "embedder_name_or_path = $embedder_name_or_path"
echo "candidate_pool = $candidate_pool"

# 设置默认值
: ${batch_size:=1024}
: ${input_file:="../data/embedder_train_eval_data/cross_validation/hn_mine_input.jsonl"}
: ${output_file:="../data/embedder_train_eval_data/cross_validation/finetune_data_hn_mined_round1.jsonl"}
: ${embedder_name_or_path:="../model_output/icl_finetune_round1/merged_model_lora_epoch_1"}
: ${candidate_pool:="../data/embedder_train_eval_data/cross_validation/corpus.jsonl"}

# 固定参数
embedder_model_class="decoder-only-icl"
pooling_method="last_token"
range_for_sampling="2-200"
negative_number=15
shuffle_data=True
query_instruction_for_retrieval="Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer."
query_instruction_format="<instruct>{}\n<query>{}"
add_examples_for_task=True
embedder_query_max_length=1024
embedder_passage_max_length=512

model_args="\
    --embedder_name_or_path $embedder_name_or_path \
    --embedder_model_class $embedder_model_class \
    --pooling_method $pooling_method \
"

data_args="\
    --input_file $input_file \
    --output_file $output_file \
    --candidate_pool $candidate_pool \
    --range_for_sampling $range_for_sampling \
    --negative_number $negative_number \
    --shuffle_data $shuffle_data \
    --query_instruction_for_retrieval \"$query_instruction_for_retrieval\" \
    --query_instruction_format '$query_instruction_format' \
    --add_examples_for_task $add_examples_for_task \
    --embedder_query_max_length $embedder_query_max_length \
    --embedder_passage_max_length $embedder_passage_max_length \
"

cmd="python hn_mine.py \
    $model_args \
    $data_args \
    --batch_size $batch_size \
    --devices cuda:0 \
"

echo $cmd
eval $cmd