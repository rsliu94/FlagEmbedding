import torch
import torch.nn.functional as F
from peft import PeftModel
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}'

def get_detailed_example(task_description: str, query: str, response: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}\n<response>{response}'

def get_new_queries(queries, query_max_len, examples_prefix, tokenizer):
    inputs = tokenizer(
        queries,
        max_length=query_max_len - len(tokenizer('<s>', add_special_tokens=False)['input_ids']) - len(
            tokenizer('\n<response></s>', add_special_tokens=False)['input_ids']),
        return_token_type_ids=False,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False
    )
    prefix_ids = tokenizer(examples_prefix, add_special_tokens=False)['input_ids']
    suffix_ids = tokenizer('\n<response>', add_special_tokens=False)['input_ids']
    new_max_length = (len(prefix_ids) + len(suffix_ids) + query_max_len + 8) // 8 * 8 + 8
    new_queries = tokenizer.batch_decode(inputs['input_ids'])
    for i in range(len(new_queries)):
        new_queries[i] = examples_prefix + new_queries[i] + '\n<response>'
    return new_max_length, new_queries

task = 'Given a web search query, retrieve relevant passages that answer the query.'
examples = [
  {'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
   'query': 'what is a virtual interface',
   'response': "A virtual interface is a software-defined abstraction that mimics the behavior and characteristics of a physical network interface. It allows multiple logical network connections to share the same physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security and management purposes."},
  {'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
   'query': 'causes of back pain in female for a week',
   'response': "Back pain in females lasting a week can stem from various factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management."}
]
examples = [get_detailed_example(e['instruct'], e['query'], e['response']) for e in examples]
examples_prefix = '\n\n'.join(examples) + '\n\n' # if there not exists any examples, just set examples_prefix = ''
queries = [
    get_detailed_instruct(task, 'how much protein should a female eat'),
    get_detailed_instruct(task, 'summit define')
]
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]
query_max_len, doc_max_len = 512, 512

# model_path = 'BAAI/bge-en-icl'
model_path = '/root/autodl-tmp/github/FlagEmbedding/projects/model_output/icl_ft_debug/checkpoint-10'
load_with_lora = True
lora_path = '/root/autodl-tmp/github/FlagEmbedding/projects/model_output/icl_ft_debug/lora'
# dir structure:
# ls /root/autodl-tmp/github/FlagEmbedding/pr
# ojects/model_output/icl_ft_debug/checkpoint-10
# README.md                  added_tokens.json  rng_state.pth            tokenizer_config.json  zero_to_fp32.py
# adapter_config.json        global_step10      special_tokens_map.json  trainer_state.json
# adapter_model.safetensors  latest             tokenizer.model          training_args.bin
# ls /root/autodl-tmp/github/FlagEmbedding/projects/model_output/icl_ft_debug/checkpoint-10/global_step10/
# mp_rank_00_model_states.pt  zero_pp_rank_0_mp_rank_00_optim_states.pt

# ls /root/autodl-tmp/github/FlagEmbedding/projects/model_output/icl_ft_debug/lora
# README.md            adapter_model.safetensors  special_tokens_map.json  tokenizer_config.json
# adapter_config.json  added_tokens.json          tokenizer.model          training_args.bin



if load_with_lora:
    print("Loading LoRA model from {}".format(lora_path))
    base_model = AutoModel.from_pretrained('BAAI/bge-en-icl')
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=False)
else:
    print("Loading standard model from {}".format(model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
model.eval()

new_query_max_len, new_queries = get_new_queries(queries, query_max_len, examples_prefix, tokenizer)

query_batch_dict = tokenizer(new_queries, max_length=new_query_max_len, padding=True, truncation=True, return_tensors='pt')
doc_batch_dict = tokenizer(documents, max_length=doc_max_len, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    query_outputs = model(**query_batch_dict)
    query_embeddings = last_token_pool(query_outputs.last_hidden_state, query_batch_dict['attention_mask'])
    doc_outputs = model(**doc_batch_dict)
    doc_embeddings = last_token_pool(doc_outputs.last_hidden_state, doc_batch_dict['attention_mask'])
    
# normalize embeddings
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
scores = query_embeddings @ doc_embeddings.T
print(scores.tolist())

# raw: [[0.5815188884735107, 0.27741539478302], [0.23050722479820251, 0.5201163291931152]]
# ckpt-10: [[0.5789058208465576, 0.26900291442871094], [0.21650180220603943, 0.5238620042800903]]
# lora : [[0.5789058208465576, 0.26900291442871094], [0.21650180220603943, 0.5238620042800903]]
