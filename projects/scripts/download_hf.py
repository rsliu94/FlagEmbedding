# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

# scp -rP 66666 /root/autodl-tmp/xxx root@region-3.autodl.com:/root/autodl-tmp/
# scp -rP 13130 projects/model_output/icl_finetune_iter1_hn/lora_epoch_2 root@connect.yza1.seetacloud.com:/root/autodl-tmp/github/FlagEmbedding/projects/model_output/icl_finetune_iter1_hn

# 在老机器上执行
# scp -rP 36485 /root/autodl-tmp/github/KddCup-2024-OAG-Challenge-1st-Solutions root@connect.bjc1.seetacloud.com:/root/autodl-tmp/github/
# scp -rP 36485 /root/autodl-tmp/github/FlagEmbedding.zip root@connect.bjc1.seetacloud.com:/root/autodl-tmp/github/
# scp -rP 36485 /root/autodl-tmp/cache/hub/models--Qwen--Qwen2.5-14B-Instruct root@connect.bjc1.seetacloud.com:/root/autodl-tmp/cache/hub/
# scp -rP 36485 /root/autodl-tmp/cache/hub/models--BAAI--bge-en-icl root@connect.bjc1.seetacloud.com:/root/autodl-tmp/cache/hub/

# Load model directly
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
# model_name = "Qwen/Qwen2.5-14B-Instruct"
model_name = "google/gemma-2-9b-it"
print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
print(f"Downloaded {model_name}.")
