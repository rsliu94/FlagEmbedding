from transformers import AutoTokenizer, AutoModel
import torch
import os
from peft import PeftModel

base_model_path = 'BAAI/bge-en-icl'
output_dir = '/root/autodl-tmp/github/FlagEmbedding/projects/model_output/icl_finetune_round1/'
lora_path = os.path.join(output_dir, 'lora_epoch_4')
merged_model_path = os.path.join(output_dir, 'merged_model')


print("Loading standard model from {}".format(base_model_path))
model = AutoModel.from_pretrained(base_model_path)

print("Loading lora model from {}".format(lora_path))
model = PeftModel.from_pretrained(model, lora_path)
print("Merging lora model with base model")
model = model.merge_and_unload()

print("Saving merged model to {}".format(merged_model_path))
model.save_pretrained(merged_model_path)

tokenizer = AutoTokenizer.from_pretrained(lora_path)
print("Saving tokenizer to {}".format(merged_model_path))
tokenizer.save_pretrained(merged_model_path)
print("Done")
