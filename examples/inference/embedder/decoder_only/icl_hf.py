import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
import numpy as np
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import faiss
import pandas as pd
from FlagEmbedding.finetune.embedder.encoder_only.base.metrics import mapk, apk, mean_average_precision_at_k, recall_at_k

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    # text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def preprocess_data(train_data, 
                    misconception_mapping, 
                    query_text_version, 
                    with_instruction=True, 
                    with_misconception=True, 
                    filter_na_misconception=True):

    # 1. Melt answer columns and create base dataframe
    answer_cols = ['AnswerAText', 'AnswerBText', 'AnswerCText', 'AnswerDText']
    answer_values = ['A', 'B', 'C', 'D']

    # Melt the answer columns
    melted_answers = pd.melt(
        train_data,
        id_vars=['QuestionId', 'QuestionText', 'ConstructId', 'ConstructName', 
                'SubjectId', 'SubjectName', 'CorrectAnswer'],
        value_vars=answer_cols,
        var_name='AnswerColumn',
        value_name='WrongAnswerText'
    )
    # Add WrongAnswer column based on AnswerColumn
    melted_answers['WrongAnswer'] = melted_answers['AnswerColumn'].map(
        dict(zip(answer_cols, answer_values))
    )


    # 2. Add MisconceptionId and MisconceptionName if with_misconception = True
    if with_misconception:
        misconception_cols = [f'Misconception{x}Id' for x in ['A', 'B', 'C', 'D']]  # Fixed column names
        melted_misconceptions = pd.melt(
            train_data,
            id_vars=['QuestionId', 'CorrectAnswer'],
            value_vars=misconception_cols,
            var_name='MisconceptionColumn',
            value_name='MisconceptionId'
        )
        melted_misconceptions['WrongAnswer'] = melted_misconceptions['MisconceptionColumn'].str[-3]
        
        df = melted_answers.merge(
            melted_misconceptions[['QuestionId', 'WrongAnswer', 'MisconceptionId']], 
            on=['QuestionId', 'WrongAnswer'], 
            how='left'
        )

        df = df.merge(
            misconception_mapping[['MisconceptionId', 'MisconceptionName']], 
            on='MisconceptionId', 
            how='left'
        )
    else:
        df = melted_answers

    # Create CorrectAnswerText column
    correct_answers = df[['QuestionId', 'WrongAnswer', 'WrongAnswerText']].copy()
    correct_answers = correct_answers[
        correct_answers['WrongAnswer'] == correct_answers['QuestionId'].map(
            train_data.set_index('QuestionId')['CorrectAnswer']
        )
    ]
    correct_answers = correct_answers.rename(
        columns={'WrongAnswerText': 'CorrectAnswerText'}
    )[['QuestionId', 'CorrectAnswerText']]
    # Merge correct answer text
    df = df.merge(correct_answers, on='QuestionId', how='left')
    # Filter out the correct answer
    df = df[df['WrongAnswer'] != df['CorrectAnswer']]
    # Create QuestionId_Answer column
    df['QuestionId_Answer'] = df['QuestionId'].astype(str) + '_' + df['WrongAnswer']
    if with_misconception:
        final_columns = ['QuestionId_Answer', 'QuestionId', 'QuestionText', 'ConstructId',
            'ConstructName', 'SubjectId', 'SubjectName', 'CorrectAnswer', 'CorrectAnswerText',
            'WrongAnswerText', 'WrongAnswer', 'MisconceptionId', 'MisconceptionName']
    else:
        final_columns = ['QuestionId_Answer', 'QuestionId', 'QuestionText', 'ConstructId',
            'ConstructName', 'SubjectId', 'SubjectName', 'CorrectAnswer', 'CorrectAnswerText',
            'WrongAnswerText', 'WrongAnswer']
    df = df[final_columns]
    
    if query_text_version == "v1":
        df["query_text"] = df["ConstructName"] + " " + df["QuestionText"] + " " + df["WrongAnswerText"]
        df["query_text"] = df["query_text"].apply(preprocess_text)
    else:
        raise ValueError(f"Invalid query_text_version: {query_text_version}")
    
    if with_instruction:
        task_description = 'Given a math question and an incorrect answer, please retrieve the most accurate reason for the misconception leading to the incorrect answer.'
        df['query_text'] = df.apply(lambda row: f"Instruction:{task_description}\nQuery:{row['query_text']}", axis=1)

    # filter out rows with NA in MisconceptionId
    if with_misconception and filter_na_misconception:
        df = df[df['MisconceptionId'].notna()]
    
    df = df.sort_values(['QuestionId', 'QuestionId_Answer']).reset_index(drop=True)
    df['order_index'] = df['QuestionId_Answer']
    
    return df

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

RAW_DATA_DIR = "/root/autodl-tmp/github/FlagEmbedding/examples/finetune/embedder/eval_data/raw_data"
EVAL_DATA_DIR = "/root/autodl-tmp/github/FlagEmbedding/examples/finetune/embedder/eval_data"
misconception_mapping = pd.read_csv(f"{RAW_DATA_DIR}/misconception_mapping.csv")
documents = misconception_mapping['MisconceptionName'].values.tolist()

task = 'Given a math question and a misconcepted incorrect answer to it, retrieve the most accurate reason for the misconception leading to the incorrect answer.'
examples = [
  {'instruct': 'Given a math question about divide decimals by 10 and a misconcepted incorrect answer to it, retrieve the most accurate reason for the misconception leading to the incorrect answer.',
   'query': '\\( 43.2 \\div 10= \\) Incorrect answer : \\( 33.2 \\)',
   'response': 'Subtracts instead of divides'},
  {'instruct': 'Given a math question about know the equation of the axes and a misconcepted incorrect answer to it, retrieve the most accurate reason for the misconception leading to the incorrect answer.',
   'query': 'What is the equation of the \\( y \\) axis? Incorrect answer : \\( y=0 \\)',
   'response': 'Confuses the equations of vertical and horizontal lines'}
]
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
examples_prefix = '\n\n'.join(examples) + '\n\n' # if there not exists any examples, just set examples_prefix = ''
# queries = [
#     get_detailed_instruct(task, 'The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.]() Incorrect answer : Not enough information'),
#     get_detailed_instruct(task, 'The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.]() Incorrect answer : obtuse')
# ]
val_data = pd.read_csv(f"{RAW_DATA_DIR}/validation_v2/val.csv")
val_preprocessed = preprocess_data(val_data, misconception_mapping, 
                            query_text_version='v1',
                            with_instruction=False, 
                            with_misconception=True, 
                            filter_na_misconception=True)
correct_ids = val_preprocessed["MisconceptionId"].values.tolist()
print(f"Number of correct ids: {len(correct_ids)}")
queries = []
for idx, row in val_preprocessed.iterrows():
    task_description = f'Given a math question about {preprocess_text(row["ConstructName"])} and a misconcepted incorrect answer to it, retrieve the most accurate reason for the misconception leading to the incorrect answer.'
    query = f'{row["QuestionText"]} \n Incorrect answer : {row["WrongAnswerText"]}'
    queries.append(get_detailed_instruct(task_description=task_description, query=query))

query_max_len, doc_max_len = 384, 128


print(f"Number of queries: {len(queries)}, number of documents: {len(documents)}")

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
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-en-icl')
model = AutoModel.from_pretrained('BAAI/bge-en-icl')

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
for doc in tqdm(documents):
    cur_doc_max_len = max(cur_doc_max_len, len(tokenizer(doc)['input_ids']))
print(f"Current document max length: {cur_doc_max_len}")


doc_embeddings = inference_doc(documents, tokenizer, model, doc_max_len, batch_size, device)
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










# tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-en-icl')
# new_query_max_len, new_queries = get_new_queries(queries, query_max_len, examples_prefix, tokenizer)

# query_batch_dict = tokenizer(new_queries, max_length=new_query_max_len, padding=True, truncation=True, return_tensors='pt')
# doc_batch_dict = tokenizer(documents, max_length=doc_max_len, padding=True, truncation=True, return_tensors='pt')

# model = AutoModel.from_pretrained('BAAI/bge-en-icl')
# model.eval()
# with torch.no_grad():
#     query_outputs = model(**query_batch_dict)
#     query_embeddings = last_token_pool(query_outputs.last_hidden_state, query_batch_dict['attention_mask'])
#     doc_outputs = model(**doc_batch_dict)
#     doc_embeddings = last_token_pool(doc_outputs.last_hidden_state, doc_batch_dict['attention_mask'])
    
# # normalize embeddings
# query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
# doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
# scores = query_embeddings @ doc_embeddings.T
# print(scores.tolist())