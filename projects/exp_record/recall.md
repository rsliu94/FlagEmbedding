## Experiment Results

### FineTune BGE-ICL



### BGE-ICL
- **Configuration**
  - Model: `BAAI/bge-en-icl`
  - bnb: 8bit+fp16; 8G VRAM
  - batch size: 32; 3min for query inference
  - adding examples in query -> better performance
- **Results**
  - On validation_v2 Recall@25: 0.57181, MAP@25: 0.2048
  <!-- - map@25_score: 0.20808355681111446,recall@25_score: 0.5782422293676313 [Kaggle notebook infer[30min]]-->
  - **LB Score: ?**
- **Variants**
  - model.half() 17G VRAM
  - batch size: 16; 3min for query inference
- **Results**
  - On validation_v2 Recall@25: 0.5814, MAP@25: 0.2156
  - **LB Score: ?**

### BGE-Small-EN-v1.5 (Query Text v1)
- **Configuration**
  - Model: `BAAI/bge-small-en-v1.5`
  - Filter NA Misconception: `True`
  - With Instruction: `False`
  - Query Text Version: `v1`
- **Results**
  <!-- - CV Recall@25: 0.4970 ± 0.0119 -->
  <!-- - CV MAP@25: 0.1707 ± 0.0040 -->
  <!-- - Validation v1: Recall@25: 0.5041, MAP@25: 0.1755 -->
  - Validation v2: Recall@25: 0.5343, MAP@25: 0.1879
  - **LB Score: 0.188**
  - Keep digits or special characters in query text: On validation_v2 Recall@25: 0.5220, MAP@25: 0.1914 [LB Score: 0.177]
  - Add 'SubjectName': 0.1758; Remove 'ConstructName': 0.1404 [fix query text version to v1 after this experiment[11/24]]

### BGE-Large-EN-v1.5
- **Configuration**
  - Model: `BAAI/bge-large-en-v1.5`
  - Filter NA Misconception: `False`
  - With Instruction: `False`
  - Query Text Version: `v1`
- **Results**
  - On validation_v2 Recall@25: 0.5874, MAP@25: 0.2015
  - **LB Score: 0.161**

- Variants:
  - Model: `/root/autodl-tmp/github/FlagEmbedding/examples/finetune/embedder/FT-1125-bge-large-en-v1.5`
  - On validation_v2 Recall@25: 0.7503, MAP@25: 0.2845
  <!-- - **LB Score: 0.197 [use same model but can enhance it by training with whole dataset]** -->
  - **LB Score: 0.222 [fine-tuned on the whole train.csv]**

  - Model: `/root/autodl-tmp/github/FlagEmbedding/examples/finetune/embedder/FT-1125-bge-large-en-v1.5-validation-v2`
  - On validation_v2 Recall@25: 0.7556, MAP@25: 0.2888
  - [larger batch size: 4 -> 16]




# Archived Experiments

### BGE-Small-EN-v1.5 (Query Text v2)
- **Configuration**
  - Model: `bge-small-en-v1.5`
  - Filter NA Misconception: `True`
  - With Instruction: `False`
  - Query Text Version: `v2`
- **Results**
  <!-- - CV Recall@25: 0.4517 ± 0.0075 -->
  <!-- - CV MAP@25: 0.1501 ± 0.0050 -->
  <!-- - Validation v1: Recall@25: 0.4685, MAP@25: 0.1510 -->
  - Validation v2: Recall@25: 0.4930, MAP@25: 0.1652
  - **LB Score: 0.168**
  - **Note**: Why v2 query text is worse than v1? [maybe, v1 has removed digits from the text.]

### BGE-Small-EN-v1.5 with Instructions
- **Configuration**
  - Model: `bge-small-en-v1.5`
  - Filter NA Misconception: `True`
  - With Instruction: `True`
  - Query Text Version: `v2`
- **Results**
  <!-- - CV Recall@25: 0.3430 ± 0.0072 -->
  <!-- - CV MAP@25: 0.1118 ± 0.0027 -->
  <!-- - Validation v1: Recall@25: 0.3473, MAP@25: 0.1124 -->
  - Validation v2: Recall@25: 0.3810, MAP@25: 0.1238
  - **LB Score: 0.117**

### SFR-Embedding-2_R
- **Configuration**
  - Model: `SFR-Embedding-2_R`
  - Status: No fine-tuning
- **Results**
  - Validation v2: Recall@25: 0.6237, MAP@25: 0.2090
  - Base LB Score: 0.205
  - After LoRA Fine-tuning LB: 0.353, Post-Fine-tuning Validation v2 MAP@25: 0.874 (Note: Data leakage because the lora may trained on my validation set)
