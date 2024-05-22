#!/bin/bash

WIKI_PATH=data/wiki
MAIN_PATH=$WIKI_PATH
WIKI_CASED_PATH=data/wiki-cased/en.train.raw.bert-base-uncased.hdf5 
BOOKCORPUS_PATH=data/bookcorpus-cased/txt/train.raw.bert-base-uncased.hdf5 
DESTINATION_PATH=data/wiki-cased/en-bc.train.raw.bert-base-uncased.hdf5

# tools paths
TOOLS_PATH=$MAIN_PATH/tools

CHUNK_SIZE=8192

python $TOOLS_PATH/concatenate_hdf5.py $WIKI_CASED_PATH $BOOKCORPUS_PATH $DESTINATION_PATH --chunk_size $CHUNK_SIZE
