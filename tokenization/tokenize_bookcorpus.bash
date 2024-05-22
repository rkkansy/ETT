DATA_DIR=data/bookcorpus-cased/
TOKENIZER=bert-base-uncased

python tokenization/tokenize_dataset.py $DATA_DIR train.raw $TOKENIZER