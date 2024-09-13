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
import matplotlib.pyplot as plt

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

def compare_dynamics(args):
    model_path_first = "/home/robert/Documents/ETT/models/BERT-small-test2"
    model_path_second = "/home/robert/Documents/ETT/models/BERT-small-test2"

    data_train_first = os.path.join(model_path_first, "instances_data.hdf5")
    data_train_second = os.path.join(model_path_second, "instances_data.hdf5")

    data_eval_first_seen = os.path.join(model_path_first, "dynamics_eval_42.hdf5")
    
    data_eval_first_unseen = os.path.join(model_path_first, "dynamics_eval_43.hdf5")
    
    instance_order_first = data_train_first['instance_order']
    instance_order_second = data_train_second['instance_order']

    for i in range(1, 11):
        mean_confidence_first_seen = data_eval_first_seen['mean_confidence'][:100*96, i]
        mean_confidence_first_unseen = data_eval_first_seen['mean_confidence'][100*96:100*96*2, i]

        mean_confidence_first_seen2 = data_eval_first_unseen['mean_confidence'][:100*96, i]
        mean_confidence_first_unseen2 = data_eval_first_unseen['mean_confidence'][100*96:100*96*2, i]
    
        # 2. Confidence scores comparison
        print(i, "Mean confidence of first 100 batches:", np.mean(mean_confidence_first_seen))
        print(i, "Mean confidence of last 100 batches:", np.mean(mean_confidence_first_unseen))
        print()
                # 2. Confidence scores comparison
        print(i, "Mean confidence of first 100 batches rdm seed:", np.mean(mean_confidence_first_seen2))
        print(i, "Mean confidence of last 100 batches rdm seed:", np.mean(mean_confidence_first_unseen2))
        print()
"""
    # Analyze top and bottom 10, 20, 30 percent
    for frac in [1, 5, 10, 20, 50]:

        print(f"\n--- Comparing top and bottom {frac}% instances ---")

        # Model 1 Seen Data
        bottom_indices, top_indices = get_percent_indices(mean_confidence_first_seen, frac)
        top_instances_first_seen = set(instance_order_first[top_indices])
        bottom_instances_first_seen = set(instance_order_first[bottom_indices])

        # Model 1 Unseen Data
        bottom_indices, top_indices = get_percent_indices(mean_confidence_first_unseen, frac)
        top_instances_first_unseen = set(instance_order_second[top_indices])
        bottom_instances_first_unseen = set(instance_order_second[bottom_indices])
        
        # Model 2 Seen Data
        bottom_indices, top_indices = get_percent_indices(mean_confidence_second_seen, frac)
        top_instances_second_seen = set(instance_order_second[top_indices])
        bottom_instances_second_seen = set(instance_order_second[bottom_indices])

        # Model 2 Unseen Data
        bottom_indices, top_indices = get_percent_indices(mean_confidence_second_unseen, frac)
        top_instances_second_unseen = set(instance_order_first[top_indices])
        bottom_instances_second_unseen = set(instance_order_first[bottom_indices])

        # Calculate and print intersections
        print(f"Intersection of top {frac}% for Model 1 training data: {len(top_instances_first_seen & top_instances_second_unseen)} / {len(top_instances_first_seen)} | {len(top_instances_first_seen & top_instances_second_unseen) / len(top_instances_first_seen)}")
        print(f"Intersection of bottom {frac}% between Model 1 Seen and Model 2 Seen: {len(bottom_instances_first_seen & bottom_instances_second_unseen)} / {len(top_instances_first_seen)} | {len(bottom_instances_first_seen & bottom_instances_second_unseen) / len(top_instances_first_seen)}")
        
        print(f"Intersection of top {frac}% between Model 1 Unseen and Model 2 Unseen: {len(top_instances_first_unseen & top_instances_second_seen)} / {len(top_instances_first_seen)} | {len(top_instances_first_unseen & top_instances_second_seen) / len(top_instances_first_seen)}")
        print(f"Intersection of bottom {frac}% between Model 1 Unseen and Model 2 Unseen: {len(bottom_instances_first_unseen & bottom_instances_second_seen)} / {len(top_instances_first_seen)} | {len(bottom_instances_first_unseen & bottom_instances_second_seen) / len(top_instances_first_seen)}")
        print()
"""


def load_log_data(directory_path):
    log_data = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            # Manually parsing each file
            with open(file_path, 'r') as file:
                lines = file.readlines()
                headers = lines[0].strip().split(', ')
                headers = [header.split('=')[0] for header in headers if '=' in header]  # Extract proper header names
                
                # Prepare a list of dictionaries for DataFrame construction
                entries = []
                for line in lines:
                    values = line.strip().split(', ')
                    # Ensure each part of the line can be split on '='
                    if all('=' in value for value in values):
                        entry = {}
                        for value in values:
                            parts = value.split('=')
                            if len(parts) == 2 and parts[0] in headers:
                                try:
                                    entry[parts[0]] = float(parts[1])  # Extract and convert data
                                except ValueError:
                                    print(f"Skipping malformed data entry in {filename}: {value}")
                        if entry:  # Ensure the entry is not empty
                            entries.append(entry)
                    else:
                        print(f"Skipping malformed line in {filename}: {line}")

                if entries:
                    df = pd.DataFrame(entries)
                    
                    # Calculate the log of evaluation perplexity
                    if 'eval_ppl' in df.columns:
                        df['log_eval_ppl'] = np.log(df['eval_ppl'])
                    
                        # Smooth the log of evaluation perplexity using a moving average
                        window_size = 10  # Define the window size for the moving average
                        df['smoothed_log_eval_ppl'] = df['log_eval_ppl'].rolling(window=window_size).mean()
                    
                        # Exponential smoothing
                        smoothing_factor = 0.2  # Alpha parameter for exponential smoothing
                        df['exp_smoothed_log_eval_ppl'] = df['log_eval_ppl'].ewm(alpha=smoothing_factor).mean()
                    
                    log_data[filename] = df
    
    return log_data

def plot_train_loss(log_data, x_limits=None, y_limits=None):
    plt.figure(figsize=(14, 7))
    
    # Plot for "one" models
    ax_one = plt.subplot(1, 2, 1)
    for name, df in log_data.items():
        if 'one' in name:
            ax_one.plot(df['train_step'], df['exp_smoothed_log_eval_ppl'], label=name.split('.')[0])
    ax_one.set_title('Training Loss for "One" Models')
    ax_one.set_xlabel('Training Step')
    ax_one.set_ylabel('Training Loss')
    ax_one.legend()
    ax_one.grid(True)

    # Set x and y limits if specified
    if x_limits:
        ax_one.set_xlim(x_limits)
    if y_limits:
        ax_one.set_ylim(y_limits)

    # Plot for "full" models
    ax_full = plt.subplot(1, 2, 2)
    for name, df in log_data.items():
        if 'full' in name:
            ax_full.plot(df['train_step'], df['exp_smoothed_log_eval_ppl'], label=name.split('.')[0])
    ax_full.set_title('Training Loss for "Full" Models')
    ax_full.set_xlabel('Training Step')
    ax_full.set_ylabel('Training Loss')
    ax_full.legend()
    ax_full.grid(True)

    # Set x and y limits if specified
    if x_limits:
        ax_full.set_xlim(x_limits)
    if y_limits:
        ax_full.set_ylim(y_limits)

    plt.tight_layout()
    plt.show()

def eval(args):
    set_seed(args)  # Added here for reproducibility
    model_path = "/project/data/models/BERT-6L-512H-10k-step-dyn"

    data_eval = load_eval_s_data_from_hdf5(os.path.join(model_path, "single_dynamics.hdf5"))

    data_train = load_train_data_from_hdf5(os.path.join(model_path, "instances_masks.hdf5"))
    # Extract relevant metrics from synchronous evaluation data
    instance_order = data_train['instance_order']

    for i in range(0, len(data_eval['mean_entropy']), 10000):
        print(i, data_eval['mean_confidence'][i])
        print(i, data_eval['mean_entropy'][i])

    confidence = data_eval['mean_confidence']
    entropy = data_eval['mean_entropy']

    mini_batch_size = 96
    gradient_acc = 16
    batch_size = mini_batch_size * gradient_acc

    sorted_confidence_batches = []
    sorted_entropy_batches = []

    total_steps = len(data_eval['mean_confidence']) // batch_size

    print(total_steps)

    for i in range(total_steps):
        sorted_confidence_batches.append(np.argsort(confidence[i * batch_size : (i + 1) * batch_size]))
        sorted_entropy_batches.append(np.argsort(entropy[i * batch_size : (i + 1) * batch_size]))

    sorted_confidence = []
    sorted_entropy = []
    for i in range(batch_size):
        for j in range(total_steps):
            sorted_confidence.append(sorted_confidence_batches[j][i] + j * batch_size)
            sorted_entropy.append(sorted_entropy_batches[j][i] + j * batch_size)

    plot_confidence_sorted = []
    plot_confidence = []
    plot_entropy_sorted = []
    plot_entropy = []

    for i in range(total_steps):
        plot_confidence_sorted.append(confidence[sorted_confidence[i * batch_size]])
        plot_entropy_sorted.append(entropy[sorted_entropy[i * batch_size]])
        
        print("confidence: " , confidence[sorted_confidence[i * batch_size]])
        print("Index: " , sorted_confidence[i * batch_size])

        print("entropy: " , entropy[sorted_entropy[i * batch_size]])
        print("Index: " , sorted_entropy[i * batch_size])
        print()
        plot_confidence.append(confidence[i * batch_size])
        plot_entropy.append(entropy[i * batch_size])

    plt.figure(figsize=(10, 6))
    plt.plot(range(total_steps), plot_confidence, label='confidence')
    plt.plot(range(total_steps), plot_confidence_sorted, label='confidence sorted')
    plt.xlabel('Steps')
    plt.ylabel('Confidence')
    plt.title('Train and Eval Log Perplexity over Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig("/project/data/models/BERT-6L-512H-10k-step-dyn/confidence") 

    plt.figure(figsize=(10, 6))
    plt.plot(range(total_steps), plot_entropy, label='entropy')
    plt.plot(range(total_steps), plot_entropy_sorted, label='entropy sorted')
    plt.xlabel('Steps')
    plt.ylabel('entropy')
    plt.title('Train and Eval Log Perplexity over Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig("/project/data/models/BERT-6L-512H-10k-step-dyn/entropy")
   
def get_percent_indices(confidences, percent):
    num_elements = int(len(confidences) * (percent / 100))
    sorted_indices = np.argsort(confidences)
    return sorted_indices[:num_elements], sorted_indices[-num_elements:]

def plot_metrics(data, step_size=1000):

    confidence = data["confidence"] 
    entropy = data["entropy"]
    variability = data["variability"] 
    correctness = data["correctness"]
    # Create a colormap and normalize based on the correctness values
    cmap = plt.cm.viridis  # You can change this to any other colormap as needed (e.g., plt.cm.coolwarm)
    norm = plt.Normalize(vmin=0, vmax=1)  # Assuming correctness ranges from 0 to 1
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Select every 1000th entry from each metric
    indices = range(0, len(confidence), step_size)
    selected_confidence = [confidence[i] for i in indices]
    selected_entropy = [entropy[i] for i in indices]
    selected_variability = [variability[i] for i in indices]
    selected_correctness = [correctness[i] for i in indices]

    # Scatter plot for Confidence over Entropy
    sc = axs[0].scatter(selected_entropy, selected_confidence, c=selected_correctness, cmap=cmap, norm=norm, alpha=0.7)
    axs[0].set_xlabel('Entropy')
    axs[0].set_ylabel('Confidence')
    axs[0].set_title('Confidence over Entropy')
    
    # Scatter plot for Confidence over Variability
    axs[1].scatter(selected_variability, selected_confidence, c=selected_correctness, cmap=cmap, norm=norm, alpha=0.7)
    axs[1].set_xlabel('Variability')
    axs[1].set_ylabel('Confidence')
    axs[1].set_title('Confidence over Variability')
    
    # Scatter plot for Entropy over Variability
    axs[2].scatter(selected_variability, selected_entropy, c=selected_correctness, cmap=cmap, norm=norm, alpha=0.7)
    axs[2].set_xlabel('Variability')
    axs[2].set_ylabel('Entropy')
    axs[2].set_title('Entropy over Variability')
    
    # Create a colorbar for the scatter plots
    cbar = fig.colorbar(sc, ax=axs.ravel().tolist(), shrink=0.95)
    cbar.set_label('Correctness')

    plt.tight_layout()
    plt.show()

def compute_dynamics(args, ckpts, mask_set):

    confidence_list = []
    entropy_list = []
    correctness_list = []
    file_path = os.path.join(args.output_dir, f"dynamics_data_{mask_set}.hdf5")

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
                print(f"correctness at checkpoint {i}: ", f[f'correctness_ckpt_{i}'][:])
                print(f"Entropy at checkpoint {i}: ", f[f'entropy_ckpt_{i}'][:])
                print(f"Confidence at checkpoint {i}: ", f[f'confidence_ckpt_{i}'][:])


    confidence = np.mean(confidence_list, axis=0)
    entropy = np.mean(entropy_list, axis=0)
    correctness = np.mean(correctness_list, axis=0)
    variability = np.std(confidence_list, axis=0)

    results = {
        "confidence": confidence,
        "entropy": entropy,
        "correctness": correctness,
        "variability": variability,
    }
    return results

def compute_partitions(args, ckpts, dynamics, mask_set, frac=0.33):

    with h5py.File(os.path.join(args.output_dir, "instance_data.hdf5"), 'r') as f:
        instance_order = f['instance_order'][:]
    
    partition_size = int(len(instance_order) * frac)
    compute_dynamics(args, ckpts, mask_set)

    sorted_ids_confidence = np.argsort(dynamics["confidence"])
    sorted_ids_variability = np.argsort(dynamics["variability"])
    sorted_ids_entropy = np.argsort(dynamics["entropy"])

    easy_instances = instance_order[sorted_ids_confidence[::-1][:partition_size]]
    hard_instances = instance_order[sorted_ids_confidence[:partition_size]]
    ambiguous_instances = instance_order[sorted_ids_variability[::-1][:partition_size]]

    high_entropy_instances = instance_order[sorted_ids_entropy[::-1][:partition_size]]
    low_entropy_instances = instance_order[sorted_ids_entropy[:partition_size]]

    partition_folder_path = os.path.join(args.output_dir, f"partitions_m-{mask_set}_{frac}")

    os.makedirs(partition_folder_path, exist_ok=True)

    with h5py.File(os.path.join(partition_folder_path, "easy.hdf5"), 'a') as f:
        f.create_dataset(f"instance_order", data=easy_instances, dtype=np.int32)
    with h5py.File(os.path.join(partition_folder_path, "hard.hdf5"), 'a') as f:
        f.create_dataset(f"instance_order", data=hard_instances, dtype=np.int32)
    with h5py.File(os.path.join(partition_folder_path, "ambiguous.hdf5"), 'a') as f:
        f.create_dataset(f"instance_order", data=ambiguous_instances, dtype=np.int32)
    with h5py.File(os.path.join(partition_folder_path, "high_entropy.hdf5"), 'a') as f:
        f.create_dataset(f"instance_order", data=high_entropy_instances, dtype=np.int32)
    with h5py.File(os.path.join(partition_folder_path, "low_entropy.hdf5"), 'a') as f:
        f.create_dataset(f"instance_order", data=low_entropy_instances, dtype=np.int32)
    
    results = {
        "easy_instances": easy_instances,
        "hard_instances": hard_instances,
        "ambiguous_instances": ambiguous_instances,
        "high_entropy_instances": high_entropy_instances,
        "low_entropy_instances": low_entropy_instances
    }
    return results

def evaluate_partitions(partition_data):
    # Extract the partition sets from the results dictionary
    easy_instances = set(partition_data['easy_instances'])
    hard_instances = set(partition_data['hard_instances'])
    ambiguous_instances = set(partition_data['ambiguous_instances'])
    high_entropy_instances = set(partition_data['high_entropy_instances'])
    low_entropy_instances = set(partition_data['low_entropy_instances'])
    
    # Calculate the intersections
    intersections = {
        'easy_hard': len(easy_instances & hard_instances),
        'easy_ambiguous': len(easy_instances & ambiguous_instances),
        'hard_ambiguous': len(hard_instances & ambiguous_instances),
        'high_entropy_low_entropy': len(high_entropy_instances & low_entropy_instances),
        'easy_high_entropy': len(easy_instances & high_entropy_instances),
        'hard_high_entropy': len(hard_instances & high_entropy_instances),
        'ambiguous_high_entropy': len(ambiguous_instances & high_entropy_instances)
    }
    
    print(intersections)

def compare_masks(args):
    # Constructing file paths
    mask_path5 = os.path.join(args.mask_path, "mask-set-1.hdf5")
    mask_path6 = os.path.join(args.mask_path, "mask-set-2.hdf5")

    # Loading the masks
    masks5 = h5py.File(mask_path5, 'r')['masks'][:]
    masks6 = h5py.File(mask_path6, 'r')['masks'][:]

    # Comparing shapes
    print("Shapes of mask sets:")
    print(f"Mask Set 5: {masks5.shape}")
    print(f"Mask Set 6: {masks6.shape}")

    # Checking if the shapes are identical
    shape_comparison = (masks5.shape == masks6.shape)
    print(f"Are all mask shapes identical? {shape_comparison}")

    # Simple content comparison example (for exact matches, which might be impractical for large data)
    if shape_comparison:
        same_5_6 = np.array_equal(masks5, masks6)
        mae = np.mean(np.abs(masks5 - masks6))
        print(f"Mean Absolute Error between Mask Set 5 and Mask Set 6: {mae}")

        print("Content comparison:")
        print(f"Masks 5 and 6 are the same: {same_5_6}")

    else:
        print("Skipping content comparison due to differing shapes.")

def main():
    parser = process_args()
    args = parser.parse_args()

    if args.compute_dynamics:
        res = compute_dynamics(args, args.dynamics_ckpts_list, args.mask_set)
        compute_partitions(args, args.dynamics_ckpts_list, res, args.mask_set, frac=0.33)

    plot_metrics(res, 100)

if __name__ == "__main__":
    main()

"""
    for i in range(2, 5):
        dynamics = compute_dynamics(args, [i], 1)
        dynamics1 = compute_dynamics(args, [i], 2)
        train_batch_size = 64
        checkpoint_amount = i*1000*train_batch_size

        print(len(dynamics["confidence"]))
        print(len(dynamics1["confidence"]))

        print(np.mean(dynamics["confidence"][:checkpoint_amount]), np.mean(dynamics1["confidence"][:checkpoint_amount]))
        print(np.mean(dynamics["entropy"][:checkpoint_amount]), np.mean(dynamics1["entropy"][:checkpoint_amount]))
        print(np.mean(dynamics["correctness"][:checkpoint_amount]), np.mean(dynamics1["correctness"][:checkpoint_amount]))
        print(np.mean(dynamics["variability"][:checkpoint_amount]), np.mean(dynamics1["variability"][:checkpoint_amount]))

        print(np.mean(dynamics["confidence"][checkpoint_amount:]), np.mean(dynamics1["confidence"][checkpoint_amount:]))
        print(np.mean(dynamics["entropy"][checkpoint_amount:]), np.mean(dynamics1["entropy"][checkpoint_amount:]))
        print(np.mean(dynamics["correctness"][checkpoint_amount:]), np.mean(dynamics1["correctness"][checkpoint_amount:]))
        print(np.mean(dynamics["variability"][checkpoint_amount:]), np.mean(dynamics1["variability"][checkpoint_amount:]))
"""