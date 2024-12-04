import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch import Tensor

def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

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

def get_new_queries_examples_list(queries, query_max_len, examples_prefix_list, tokenizer):
    assert len(examples_prefix_list) == len(queries)
    inputs = tokenizer(
        queries,
        max_length=query_max_len - len(tokenizer('<s>', add_special_tokens=False)['input_ids']) - len(
            tokenizer('\n<response></s>', add_special_tokens=False)['input_ids']),
        return_token_type_ids=False,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False
    )
    # print(f"DEBUG: inputs max length: {max(len(x) for x in inputs['input_ids'])}")
    prefix_ids_list = tokenizer(examples_prefix_list, add_special_tokens=False)['input_ids']
    suffix_ids = tokenizer('\n<response>', add_special_tokens=False)['input_ids']
    new_max_length = (max(len(prefix_ids) for prefix_ids in prefix_ids_list) + len(suffix_ids) + query_max_len + 8) // 8 * 8 + 8
    # print(f"New max length: {new_max_length}")
    new_queries = tokenizer.batch_decode(inputs['input_ids'])
    for i in range(len(new_queries)):
        new_queries[i] = examples_prefix_list[i] + new_queries[i] + '\n<response>'
    return new_max_length, new_queries

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

@torch.no_grad()
def inference_query_examples_list(queries, query_max_len, examples_prefix_list, tokenizer, model, batch_size, device):
    print("Getting query embeddings...")
    query_embeddings = []
    for i in tqdm(range(0, len(queries), batch_size)):
        batch = queries[i:i+batch_size]
        batch_examples_prefix = examples_prefix_list[i:i+batch_size]
        new_max_length, new_queries = get_new_queries_examples_list(batch, query_max_len, batch_examples_prefix, tokenizer)
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