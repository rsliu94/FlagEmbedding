# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

# Load model directly
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("BAAI/bge-reranker-v2-minicpm-layerwise", trust_remote_code=True)