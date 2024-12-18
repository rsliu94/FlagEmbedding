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
        if hasattr(model, 'encode'):
            embedding = model.encode(batch_dict)
        else:
            outputs = model(**batch_dict)
            embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)
        embedding = embedding.detach().cpu().numpy()
        doc_embeddings.append(embedding)
    doc_embeddings = np.concatenate(doc_embeddings, axis=0)
    print(f"Document embeddings shape: {doc_embeddings.shape}")
    return doc_embeddings

@torch.no_grad()
def inference_query_base(queries, tokenizer, model, query_max_len, batch_size, device):
    print("Getting query embeddings...")
    query_embeddings = []
    for i in tqdm(range(0, len(queries), batch_size)):
        batch = queries[i:i+batch_size]
        batch_dict = tokenizer(batch, max_length=query_max_len, padding=True, truncation=True, return_tensors='pt')
        batch_dict = batch_to_device(batch_dict, device)
        if hasattr(model, 'encode'):
            embedding = model.encode(batch_dict)
        else:
            outputs = model(**batch_dict)
            embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)
        embedding = embedding.detach().cpu().numpy()
        query_embeddings.append(embedding)
    query_embeddings = np.concatenate(query_embeddings, axis=0)
    print(f"Query embeddings shape: {query_embeddings.shape}")
    return query_embeddings

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
    prefix_ids_list = tokenizer(examples_prefix_list, add_special_tokens=False)['input_ids']
    suffix_ids = tokenizer('\n<response>', add_special_tokens=False)['input_ids']
    new_max_length = (max(len(prefix_ids) for prefix_ids in prefix_ids_list) + len(suffix_ids) + query_max_len + 8) // 8 * 8 + 8
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
        if hasattr(model, 'encode'):
            embedding = model.encode(batch_dict)
        else:
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
    print("DEBUG: checking new_queries token statistics")
    token_count = []
    for i in tqdm(range(0, len(queries), batch_size)):
        batch = queries[i:i+batch_size]
        batch_examples_prefix = examples_prefix_list[i:i+batch_size]
        new_max_length, new_queries = get_new_queries_examples_list(batch, query_max_len, batch_examples_prefix, tokenizer)
        for query in new_queries:
            token_count.append(len(tokenizer(query, add_special_tokens=False)['input_ids']))
    print(f"Token count statistics: mean={np.mean(token_count):.1f}, max={np.max(token_count)}, min={np.min(token_count)}, "
          f"25th={np.percentile(token_count, 25):.1f}, 50th={np.percentile(token_count, 50):.1f}, 75th={np.percentile(token_count, 75):.1f}, "
          f"90th={np.percentile(token_count, 90):.1f}, 95th={np.percentile(token_count, 95):.1f}, 99th={np.percentile(token_count, 99):.1f}")
        
    print("Getting query embeddings...")
    query_embeddings = []
    for i in tqdm(range(0, len(queries), batch_size)):
        batch = queries[i:i+batch_size]
        batch_examples_prefix = examples_prefix_list[i:i+batch_size]
        new_max_length, new_queries = get_new_queries_examples_list(batch, query_max_len, batch_examples_prefix, tokenizer)
        batch_dict = tokenizer(new_queries, max_length=new_max_length, padding=True, truncation=True, return_tensors='pt')
        batch_dict = batch_to_device(batch_dict, device)
        if hasattr(model, 'encode'):
            embedding = model.encode(batch_dict)
        else:
            outputs = model(**batch_dict)
            embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)
        embedding = embedding.detach().cpu().numpy()
        query_embeddings.append(embedding)
    query_embeddings = np.concatenate(query_embeddings, axis=0)
    print(f"Query embeddings shape: {query_embeddings.shape}")
    return query_embeddings


def get_inputs(pairs, tokenizer, prompt=None, query_max_len=512, doc_max_len=128):
    if prompt is None:
        prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    sep = "\n"
    prompt_inputs = tokenizer(prompt,
                              return_tensors=None,
                              add_special_tokens=False)['input_ids']
    sep_inputs = tokenizer(sep,
                           return_tensors=None,
                           add_special_tokens=False)['input_ids']
    inputs = []
    for query, passage in pairs:
        query_inputs = tokenizer(f'A: {query}',
                                 return_tensors=None,
                                 add_special_tokens=False,
                                 max_length=query_max_len,
                                 truncation=True)
        passage_inputs = tokenizer(f'B: {passage}',
                                   return_tensors=None,
                                   add_special_tokens=False,
                                   max_length=doc_max_len,
                                   truncation=True)
        
        if tokenizer.bos_token_id is not None and tokenizer.bos_token_id != tokenizer.pad_token_id:
            item = tokenizer.prepare_for_model(
                [tokenizer.bos_token_id] + query_inputs['input_ids'],
                sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=query_max_len + doc_max_len,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
        else:
            item = tokenizer.prepare_for_model(
                query_inputs['input_ids'],
                sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=query_max_len + doc_max_len,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        inputs.append(item)
    return tokenizer.pad(
            inputs,
            padding=True,
            max_length=query_max_len + doc_max_len + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
    )
