# Generate validation data
# validation data do not modify original dataset, just split the train.csv into train/val.csv under different method:
# v1: random select 20% QuestionId from train.csv as validation set
python prepare_val_data.py