import json
import random
# seed
random.seed(42)
path = './finetune_data_minedHN.jsonl'

lines = []
with open(path, 'r') as f:
    for line in f:
        lines.append(json.loads(line))

# shuffle data and save to new file
random.shuffle(lines)

with open('./finetune_data_minedHN.jsonl', 'w') as f:
    for line in lines:
        f.write(json.dumps(line) + '\n')

print(len(lines))