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
from scipy.interpolate import interp1d

import gc
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

import numpy as np
import h5py
import pandas as pd
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
from data import *
from param import process_args
from model import SimpleBertForMaskedLM

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from ligo import initialize_model_with_ligo
from run_lm_distributed import set_seed, load_and_cache_examples
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, SimpleBertForMaskedLM, BertTokenizer)
}

def compute_dynamics(args, ckpts, mask_set):

    confidence_list = []
    entropy_list = []
    correctness_list = []
    file_path = os.path.join(args.output_dir, f"dynamics_data_{mask_set}.hdf5")

    if args.verbose:
        with h5py.File(file_path, 'r') as file:
            print("Datasets in the HDF5 file:")
            for name in file.keys():
                print(name)

    for i in ckpts:
        with h5py.File(file_path, 'r') as f:
            confidence_list.append(f[f'confidence_ckpt_{i}'][:])
            entropy_list.append(f[f'entropy_ckpt_{i}'][:])
            correctness_list.append(f[f'correctness_ckpt_{i}'][:])
            if args.verbose:
                print()
                print(f"Length of datasets: {len(f[f'correctness_ckpt_{i}'][:])}")
                print(f"Correctness at checkpoint {i}: ", f[f'correctness_ckpt_{i}'][:])
                print(f"Entropy at checkpoint {i}: ", f[f'entropy_ckpt_{i}'][:])
                print(f"Confidence at checkpoint {i}: ", f[f'confidence_ckpt_{i}'][:])

    confidence = np.mean(confidence_list, axis=0)
    entropy = np.mean(entropy_list, axis=0)
    correctness = np.mean(correctness_list, axis=0)
    confidence_variability = np.std(confidence_list, axis=0)
    entropy_variability = np.std(entropy_list, axis=0)

    results = {
        "confidence": confidence,
        "entropy": entropy,
        "correctness": correctness,
        "confidence_variability": confidence_variability,
        "entropy_variability": entropy_variability,
    }
    return results

def compute_partitions(args, dynamics, mask_set, frac=0.33):

    with h5py.File(os.path.join(args.output_dir, "instance_data.hdf5"), 'r') as f:
        instance_order = f['instance_order'][:]
    
    partition_size = int(len(instance_order) * frac)

    sorted_ids_confidence = np.argsort(dynamics["confidence"])
    sorted_ids_confidence_variability = np.argsort(dynamics["confidence_variability"])
    sorted_ids_entropy_variability = np.argsort(dynamics["entropy_variability"])

    sorted_ids_entropy = np.argsort(dynamics["entropy"])

    easy_instances = instance_order[sorted_ids_confidence[::-1][:partition_size]]
    hard_instances = instance_order[sorted_ids_confidence[:partition_size]]
    confidence_ambiguous_instances = instance_order[sorted_ids_confidence_variability[::-1][:partition_size]]
    entropy_ambiguous_instances = instance_order[sorted_ids_entropy_variability[::-1][:partition_size]]

    high_entropy_instances = instance_order[sorted_ids_entropy[::-1][:partition_size]]
    low_entropy_instances = instance_order[sorted_ids_entropy[:partition_size]]

    partition_folder_path = os.path.join(args.output_dir, f"partitions_m-{mask_set}_{frac}")

    os.makedirs(partition_folder_path, exist_ok=True)

    with h5py.File(os.path.join(partition_folder_path, "easy.hdf5"), 'a') as f:
        f.create_dataset(f"instance_order", data=easy_instances, dtype=np.int32)
    with h5py.File(os.path.join(partition_folder_path, "hard.hdf5"), 'a') as f:
        f.create_dataset(f"instance_order", data=hard_instances, dtype=np.int32)
    with h5py.File(os.path.join(partition_folder_path, "ambiguous.hdf5"), 'a') as f:
        f.create_dataset(f"instance_order", data=confidence_ambiguous_instances, dtype=np.int32)
    with h5py.File(os.path.join(partition_folder_path, "entropy_ambiguous.hdf5"), 'a') as f:
        f.create_dataset(f"instance_order", data=entropy_ambiguous_instances, dtype=np.int32)
    with h5py.File(os.path.join(partition_folder_path, "high_entropy.hdf5"), 'a') as f:
        f.create_dataset(f"instance_order", data=high_entropy_instances, dtype=np.int32)
    with h5py.File(os.path.join(partition_folder_path, "low_entropy.hdf5"), 'a') as f:
        f.create_dataset(f"instance_order", data=low_entropy_instances, dtype=np.int32)
    
    results = {
        "easy_instances": easy_instances,
        "hard_instances": hard_instances,
        "ambiguous_instances": confidence_ambiguous_instances,
        "entropy_ambiguous_instances": entropy_ambiguous_instances,
        "high_entropy_instances": high_entropy_instances,
        "low_entropy_instances": low_entropy_instances
    }
    return results
    for i in ckpts:
        dynamics = compute_dynamics(args, [i], 1)  # Dynamics for mask set 1 (used in training)
        dynamics1 = compute_dynamics(args, [i], 2)  # Dynamics for mask set 2 (not used in training)
        checkpoint_amount = i * args.gradient_accumulation_steps * args.train_batch_size * args.ckpt_steps

        print(f'Number of data points in dynamics (mask set 1) at checkpoint {i}: {len(dynamics["confidence"])}')
        print(f'Number of data points in dynamics1 (mask set 2) at checkpoint {i}: {len(dynamics1["confidence"])}')

        confidence_diff_before = np.mean(dynamics["confidence"][:checkpoint_amount]) - np.mean(dynamics1["confidence"][:checkpoint_amount])
        entropy_diff_before = np.mean(dynamics["entropy"][:checkpoint_amount]) - np.mean(dynamics1["entropy"][:checkpoint_amount])
        correctness_diff_before = np.mean(dynamics["correctness"][:checkpoint_amount]) - np.mean(dynamics1["correctness"][:checkpoint_amount])
        variability_diff_before = np.mean(dynamics["confidence_variability"][:checkpoint_amount]) - np.mean(dynamics1["confidence_variability"][:checkpoint_amount])

        confidence_diff_after = np.mean(dynamics["confidence"][checkpoint_amount:]) - np.mean(dynamics1["confidence"][checkpoint_amount:])
        entropy_diff_after = np.mean(dynamics["entropy"][checkpoint_amount:]) - np.mean(dynamics1["entropy"][checkpoint_amount:])
        correctness_diff_after = np.mean(dynamics["correctness"][checkpoint_amount:]) - np.mean(dynamics1["correctness"][checkpoint_amount:])
        variability_diff_after = np.mean(dynamics["confidence_variability"][checkpoint_amount:]) - np.mean(dynamics1["confidence_variability"][checkpoint_amount:])

        confidence_diff_m2 = np.mean(dynamics1["confidence"][:checkpoint_amount]) - np.mean(dynamics1["confidence"][checkpoint_amount:])
        confidence_diff_m1 = np.mean(dynamics["confidence"][:checkpoint_amount]) - np.mean(dynamics["confidence"][checkpoint_amount:])
        print()
        print(f'Average confidence before checkpoint {i} (mask set 1 vs. mask set 2): {np.mean(dynamics["confidence"][:checkpoint_amount])} vs. {np.mean(dynamics1["confidence"][:checkpoint_amount])} (diff: {confidence_diff_before})')
        print(f'Average confidence after checkpoint {i} (mask set 1 vs. mask set 2): {np.mean(dynamics["confidence"][checkpoint_amount:])} vs. {np.mean(dynamics1["confidence"][checkpoint_amount:])} (diff: {confidence_diff_after})')
        print()
        print(f'Average entropy before checkpoint {i} (mask set 1 vs. mask set 2): {np.mean(dynamics["entropy"][:checkpoint_amount])} vs. {np.mean(dynamics1["entropy"][:checkpoint_amount])} (diff: {entropy_diff_before})')
        print(f'Average entropy after checkpoint {i} (mask set 1 vs. mask set 2): {np.mean(dynamics["entropy"][checkpoint_amount:])} vs. {np.mean(dynamics1["entropy"][checkpoint_amount:])} (diff: {entropy_diff_after})')
        print()
        print(f'Average correctness before checkpoint {i} (mask set 1 vs. mask set 2): {np.mean(dynamics["correctness"][:checkpoint_amount])} vs. {np.mean(dynamics1["correctness"][:checkpoint_amount])} (diff: {correctness_diff_before})')
        print(f'Average correctness after checkpoint {i} (mask set 1 vs. mask set 2): {np.mean(dynamics["correctness"][checkpoint_amount:])} vs. {np.mean(dynamics1["correctness"][checkpoint_amount:])} (diff: {correctness_diff_after})')
        print()
        print(f'Average variability before checkpoint {i} (mask set 1 vs. mask set 2): {np.mean(dynamics["confidence_variability"][:checkpoint_amount])} vs. {np.mean(dynamics1["confidence_variability"][:checkpoint_amount])} (diff: {variability_diff_before})')
        print(f'Average variability after checkpoint {i} (mask set 1 vs. mask set 2): {np.mean(dynamics["confidence_variability"][checkpoint_amount:])} vs. {np.mean(dynamics1["confidence_variability"][checkpoint_amount:])} (diff: {variability_diff_after})')
        print()
        print(f'Average confidence before checkpoint {i} vs after checkpoint (mask set 1): {np.mean(dynamics["confidence"][:checkpoint_amount])} vs. {np.mean(dynamics["confidence"][checkpoint_amount:])} (diff: {confidence_diff_m1})')
        print(f'Average confidence before checkpoint {i} vs after checkpoint (mask set 2): {np.mean(dynamics1["confidence"][:checkpoint_amount])} vs. {np.mean(dynamics1["confidence"][checkpoint_amount:])} (diff: {confidence_diff_m2})')

def rearrange_data(original_data, new_order):
    keys = list(original_data.keys())  # Extract keys to a list
    new_data = {keys[i]: original_data[keys[i]] for i in new_order}
    return new_data

def main():
    parser = process_args()
    args = parser.parse_args()

    if args.compute_dynamics:
        res = compute_dynamics(args, args.dynamics_ckpts_list, args.mask_set)
        compute_partitions(args, res, args.mask_set, frac=args.partition_frac)



if __name__ == "__main__":
    main()
