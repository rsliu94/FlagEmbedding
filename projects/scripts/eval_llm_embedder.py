import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
import numpy as np
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import faiss
import pandas as pd
from FlagEmbedding.utils.metrics import mapk, apk, mean_average_precision_at_k, recall_at_k
from FlagEmbedding.utils.data_utils import preprocess_text, preprocess_data
from FlagEmbedding.utils.env_utils import get_env_info
import argparse
import json

# argparser
parser = argparse.ArgumentParser(description="Evaluate the raw LLM embedder")
parser.add_argument("--use_examples_in_query", type=bool, default=False, help="Whether to use the embedder eval data")
parser.add_argument("--validation_version", type=str, default="v2", help="The version of the validation data")
parser.add_argument("--model_path", type=str, default="BAAI/bge-en-icl", help="The path of the model")
args = parser.parse_args()
# show args
print("\nScript arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print() 

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

def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

if __name__ == "__main__":
    env_name, PROJECT_ROOT = get_env_info()
    EVAL_DATA_DIR = f"{PROJECT_ROOT}/projects/data/embedder_eval_data"
    corpus_path = f"{EVAL_DATA_DIR}/corpus.jsonl"
    queries_path = f"{EVAL_DATA_DIR}/queries_{args.validation_version}.jsonl"
    examples_path = f"{EVAL_DATA_DIR}/examples_v1.json"

    task = 'Given a math question and a misconcepted incorrect answer to it, retrieve the most accurate reason for the misconception leading to the incorrect answer.'
    with open(examples_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    # documents = ['Does not know that angles in a triangle sum to 180 degrees',
    #             'Uses dividing fractions method for multiplying fractions'
    #             ]

    @torch.no_grad()
    def inference_doc(documents, tokenizer, model, doc_max_len, batch_size, device):
        doc_embeddings = []
        print("Getting document embeddings...")
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i+batch_size]
            batch_dict = tokenizer(batch, max_length=doc_max_len, padding=True, truncation=True, return_tensors='pt')
            batch_dict = batch_to_device(batch_dict, device)
            outputs = model(**batch_dict)
            embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)
            embedding = embedding.detach().cpu().numpy()
            doc_embeddings.append(embedding)
        doc_embeddings = np.concatenate(doc_embeddings, axis=0)
        print(f"Document embeddings shape: {doc_embeddings.shape}")
        return doc_embeddings

    @torch.no_grad()
    def inference_query(queries, query_max_len, examples_prefix, tokenizer, model, batch_size, device):
        print("Getting query embeddings...")
        query_embeddings = []
        for i in tqdm(range(0, len(queries), batch_size)):
            batch = queries[i:i+batch_size]
            new_max_length, new_queries = get_new_queries(batch, query_max_len, examples_prefix, tokenizer)
            batch_dict = tokenizer(new_queries, max_length=new_max_length, padding=True, truncation=True, return_tensors='pt')
            batch_dict = batch_to_device(batch_dict, device)
            outputs = model(**batch_dict)
            embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)
            embedding = embedding.detach().cpu().numpy()
            query_embeddings.append(embedding)
        query_embeddings = np.concatenate(query_embeddings, axis=0)
        print(f"Query embeddings shape: {query_embeddings.shape}")
        return query_embeddings

    examples = [get_detailed_example(e['instruct'], e['query'], e['response']) for e in examples]
    
    if args.use_examples_in_query:
        examples_prefix = '\n\n'.join(examples) + '\n\n' # if there not exists any examples, just set examples_prefix = ''
    else:
        examples_prefix = ''
        
    # queries = [
    #     get_detailed_instruct(task, 'The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.]() Incorrect answer : Not enough information'),
    #     get_detailed_instruct(task, 'The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.]() Incorrect answer : obtuse')
    # ]
    
    corpus = [json.loads(line)['text'] for line in open(corpus_path, 'r')] # list of strings
    print(f"Number of corpus: {len(corpus)}")
    correct_ids = [json.loads(line)['correct_id'] for line in open(queries_path, 'r')] # list of floats
    print(f"Number of correct ids: {len(correct_ids)}")
    
    queries = []
    with open(queries_path, 'r') as f:
        for line in f:
            row = json.loads(line)
            task_description = f'Given a math question about {preprocess_text(row["construct_name"])} and a misconcepted incorrect answer to it, retrieve the most accurate reason for the misconception leading to the incorrect answer.'
            query = f'{row["question"]} \n Incorrect answer : {row["wrong_answer"]}'
            queries.append(get_detailed_instruct(task_description=task_description, query=query))
    print(f"Number of queries: {len(queries)}")

    query_max_len, doc_max_len = 384, 128

    print("Loading tokenizer and model...")
    # bnb_config = BitsAndBytesConfig(
    #             load_in_4bit=True,
    #             bnb_4bit_use_double_quant=True,
    #             bnb_4bit_quant_type="nf4",
    #             bnb_4bit_compute_dtype=torch.bfloat16
    #         )
    # bnb_config = BitsAndBytesConfig(
    #             load_in_4bit=False,
    #             load_in_8bit=False,
    #             bnb_4bit_compute_dtype=torch.float16,
    #             llm_int8_has_fp16_weight=True,
    #         )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.half()
    model.to(device)
    model.eval()
    batch_size = 16
    print(f"Model device: {next(model.parameters()).device}")

    print("Check query/document token length...")
    cur_query_max_len = 0
    for query in tqdm(queries):
        cur_query_max_len = max(cur_query_max_len, len(tokenizer(query)['input_ids']))
    print(f"Current query max length: {cur_query_max_len}")

    cur_doc_max_len = 0
    for doc in tqdm(corpus):
        cur_doc_max_len = max(cur_doc_max_len, len(tokenizer(doc)['input_ids']))
    print(f"Current document max length: {cur_doc_max_len}")


    doc_embeddings = inference_doc(corpus, tokenizer, model, doc_max_len, batch_size, device)
    query_embeddings = inference_query(queries, query_max_len, examples_prefix, tokenizer, model, batch_size, device)
    print("Building index...")
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    distances, indices = index.search(query_embeddings, k=25)
    print(f"Distances shape: {distances.shape}, Indices shape: {indices.shape}")

    mapk_score = mean_average_precision_at_k(correct_ids, indices, 25)
    print(f"map@25_score: {mapk_score}")

    recall_score = recall_at_k(correct_ids, indices, 25)
    print(f"recall@25_score: {recall_score}")
