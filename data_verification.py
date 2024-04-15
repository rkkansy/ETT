# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import json
import logging
import os
import pickle
import random
import re
import shutil
import sys
from typing import Dict, List, Tuple
from datetime import datetime
import time 
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from data import CoLDataset
from param import process_args
from model import SimpleBertForMaskedLM, SimpleRobertaForMaskedLM

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from ligo import initialize_model_with_ligo

import h5py
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "bert": (BertConfig, SimpleBertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, SimpleRobertaForMaskedLM, RobertaTokenizer),
}


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return CoLDataset(file_path, args.tokenizer_name, tokenizer, args.block_size,
                        split_sent=args.split_sent,
                        verbose=(args.gpu == 0))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def decode_tokens_with_mask(tokenizer, inputs, masked_indices):

    tokens = tokenizer.convert_ids_to_tokens(inputs.tolist())
    for idx, mask in enumerate(masked_indices):
        if mask:
            tokens[idx] = '[MASK]'
    return tokenizer.convert_tokens_to_string(tokens)


def verify_labels_with_instance(dataset, instance_id, expected_labels_cpu):

    retrieved_instance_tokens, _ = dataset[instance_id]
    retrieved_tokens_list = retrieved_instance_tokens.tolist()

    for idx, label in enumerate(expected_labels_cpu):
        if label != -100:
            if label != retrieved_tokens_list[idx]:
                return False  # Found a mismatch
                
    return True  # All non -100 labels matched

def load_data_in_chunks(file_path, dataset_name, chunk_size=1024):
    chunks = []
    
    with h5py.File(file_path, 'r') as f:
        dataset = f[dataset_name]
        for start in range(0, dataset.shape[0], chunk_size):
            end = min(start + chunk_size, dataset.shape[0])
            chunks.append(dataset[start:end])
    
    return chunks

def load_data_from_hdf5(file_path):
                         
    data = {
        "masks" :[],
        "mean_confidence" : [],
        "geom_mean_confidence" : [],
        "correctness" : [],
        "instance_order" : []
    }
    with h5py.File(file_path, 'r') as f:
        data['masks'] = f['masks'][:]
        data['mean_confidence'] = f['mean_confidence'][:]
        data['geom_mean_confidence'] = f['geom_mean_confidence'][:]
        data['correctness'] = f['correctness'][:]   
        data['instance_order'] = f['instance_order'][:]
    
    return data

def epoch_mean(instance_order, vals, selected_epochs, instance_range):

    instance_indices = np.array(instance_order)[instance_range[0]:instance_range[1]]
    epoch_indices = np.array(selected_epochs)
    selected_vals = vals[instance_indices[:, None], epoch_indices]
    means = np.mean(selected_vals, axis=1)

    return means

def epoch_variability(instance_order, vals, selected_epochs, instance_range):

    instance_indices = np.array(instance_order)[instance_range[0]:instance_range[1]]
    epoch_indices = np.array(selected_epochs)
    selected_vals = vals[instance_indices[:, None], epoch_indices]
    variabilities = np.var(selected_vals, axis=1)
    
    return variabilities
    

def make_plot_epoch(instance_order, masks, correctness, mean_confidence, geom_mean_confidence, instance_range, selected_epochs):

    mean_confidences = epoch_mean(instance_order, mean_confidence, selected_epochs, instance_range)
    geom_mean_confidences = epoch_mean(instance_order, geom_mean_confidence, selected_epochs, instance_range)
    
    variabilities_mean = epoch_variability(instance_order, mean_confidence, selected_epochs, instance_range)
    variabilities_geom_mean = epoch_variability(instance_order, geom_mean_confidence, selected_epochs, instance_range)
    
    correctness_means = epoch_mean(instance_order, correctness, selected_epochs, instance_range)
    correctness_colors = [plt.cm.coolwarm(x) for x in correctness_means]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))

    sc1 = axes[0].scatter(variabilities_mean, mean_confidences, c=correctness_colors, alpha=0.5)
    axes[0].set_title('New: Mean Confidences vs. Variabilities')
    axes[0].set_xlabel('Variability')
    axes[0].set_ylabel('Mean Confidence')
    axes[0].grid(True)

    # New method - Geometric Mean Confidences vs. Variabilities
    sc2 = axes[1].scatter(variabilities_geom_mean, geom_mean_confidences, c=correctness_colors, alpha=0.5)
    axes[1].set_title('New: Geometric Mean Confidences vs. Variabilities')
    axes[1].set_xlabel('Variability')
    axes[1].set_ylabel('Geometric Mean Confidence')
    axes[1].grid(True)

    # Colorbars for the scatter plots
    cbar1 = plt.colorbar(sc1, ax=axes[0])
    cbar1.set_label('Correctness')
    cbar2 = plt.colorbar(sc2, ax=axes[1])
    cbar2.set_label('Correctness')

    plt.tight_layout()
    plt.show()

            
def eval(args):
    set_seed(args)  # Added here for reproducibility

    args.train_batch_size = args.per_gpu_train_batch_size
        
    data = load_data_from_hdf5("data/dynamics/bert-6L-25k-5e.hdf5")

    print(data.keys())
    masks = data['masks']
    mean_confidence = data['mean_confidence']
    geom_mean_confidence = data['geom_mean_confidence']
    correctness = data['correctness']
    instance_order = data['instance_order']

    #epoch_mean(instance_order, mean_confidence, [1, 3, 4], [0, 10])
    epochs = [1, 2, 3]
    selected_instances = [64000, 128000]
    make_plot_epoch(instance_order, masks, correctness, mean_confidence, geom_mean_confidence, selected_instances, epochs)
    

def main():
    parser = process_args()
    args = parser.parse_args()
    eval(args)


if __name__ == "__main__":
    main()
