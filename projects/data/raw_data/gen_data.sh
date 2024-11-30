# Generate validation data
# validation data do not modify original dataset, just split the train.csv into train/val.csv under different method:
# v1: random select 33.3% QuestionId from train.csv as validation set
# v2: random select 33.3% QuestionId [with no NaN MisconceptionId] from train.csv as validation set
mkdir validation_v1 validation_v2
python prepare_val_data.py