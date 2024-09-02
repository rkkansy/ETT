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

def epoch_mean(instance_order, vals, selected_epochs, instance_range, step_size):

    instance_indices = np.array(range(instance_range[0], instance_range[1], step_size))
    epoch_indices = np.array(selected_epochs)
    selected_vals = vals[instance_indices[:, None], epoch_indices]
    means = np.mean(selected_vals, axis=1)

    return means

def epoch_variability(instance_order, vals, selected_epochs, instance_range, step_size):

    instance_indices = np.array(range(instance_range[0], instance_range[1], step_size))
    epoch_indices = np.array(selected_epochs)
    selected_vals = vals[instance_indices[:, None], epoch_indices]
    variabilities = np.var(selected_vals, axis=1)
    
    return variabilities

def select_top_indices(data, frac=0.33):

    num_to_select = int(len(data) * frac)
    sorted_indices = np.argsort(data)[::-1]
    top_indices = sorted_indices[:num_to_select]

    return top_indices

def select_bottom_indices(data, frac=0.33):
    num_to_select = int(len(data) * frac)
    sorted_indices = np.argsort(data)
    bottom_indices = sorted_indices[:num_to_select]

    return bottom_indices

def compare_indices(indices1, indices2):

    set1 = set(indices1)
    set2 = set(indices2)
    
    common_indices = set1 & set2
    unique_indices1 = set1 - set2
    unique_indices2 = set2 - set1
    
    percent_common = len(common_indices) / len(set1)
    
    return {
        "percent_common": percent_common,
        "common_indices": common_indices,
        "unique_indices1": unique_indices1,
        "unique_indices2": unique_indices2,
    }

def make_plot_epoch(args, instance_order, correctness_means1, correctness_means2, mean_confidences, geom_mean_confidences, variabilities_mean, variabilities_geom_mean):
    markers = []
    for i, corr in enumerate(correctness_means1):
        if 0 < corr <= 0.1:
            markers.append('o')  # Circle
        elif 0.1 < corr <= 0.2:
            markers.append('*')  # Star
        elif 0.2 < corr <= 0.3:
            markers.append('+')  # Plus
        elif 0.3 < corr <= 0.5:
            markers.append('x')  # Cross
        elif 0.5 < corr <= 0.7:
            markers.append('D')  # Diamond
        elif 0.7 < corr <= 0.8:
            markers.append('s')  # Square
        elif 0.8 < corr <= 1.0:
            markers.append('^')  # Triangle
        else:
            markers.append('.')  # Point

    markers2 = []
    for i, corr in enumerate(correctness_means2):
        if 0 < corr <= 0.1:
            markers2.append('o')  # Circle
        elif 0.1 < corr <= 0.2:
            markers2.append('*')  # Star
        elif 0.2 < corr <= 0.3:
            markers2.append('+')  # Plus
        elif 0.3 < corr <= 0.5:
            markers2.append('x')  # Cross
        elif 0.5 < corr <= 0.7:
            markers2.append('D')  # Diamond
        elif 0.7 < corr <= 0.8:
            markers2.append('s')  # Square
        elif 0.8 < corr <= 1.0:
            markers2.append('^')  # Triangle
        else:
            markers2.append('.')  # Point

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))

    # Plotting for Mean Confidences vs. Variabilities
    colors1 = plt.cm.coolwarm(correctness_means1)  # Precompute all colors

    colors2 = plt.cm.coolwarm(correctness_means2)  # Precompute all colors

    for x, y, m, c in zip(variabilities_mean, mean_confidences, markers, colors1):
        axes[0].scatter(x, y, marker=m, color=c, alpha=0.5)
    axes[0].set_title('New: Mean Confidences vs. Variabilities')
    axes[0].set_xlabel('Variability')
    axes[0].set_ylabel('Mean Confidence')
    axes[0].grid(True)

    # Plotting for Geometric Mean Confidences vs. Variabilities
    for x, y, m, c in zip(variabilities_geom_mean, geom_mean_confidences, markers2, colors2):
        axes[1].scatter(x, y, marker=m, color=c, alpha=0.5)
    axes[1].set_title('New: Geometric Mean Confidences vs. Variabilities')
    axes[1].set_xlabel('Variability')
    axes[1].set_ylabel('Geometric Mean Confidence')
    axes[1].grid(True)

    # Colorbars for the scatter plots
    cbar1 = plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), ax=axes[0])
    cbar1.set_label('Correctness')
    cbar2 = plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), ax=axes[1])
    cbar2.set_label('Correctness')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "epoch_plot.png"))
            
def make_plot_epoch2(args, instance_order, correctness_means, mean_confidences, geom_mean_confidences, variabilities_mean, variabilities_geom_mean):
    # Create marker array based on correctness levels
    markers = ['o' if 0 < x <= 0.1 else '*' if 0.1 < x <= 0.2 else '+' if 0.2 < x <= 0.3 else 'x' if 0.3 < x <= 0.5 else 
               'D' if 0.5 < x <= 0.7 else 's' if 0.7 < x <= 0.8 else '^' if 0.8 < x <= 1.0 else '.' for x in correctness_means]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))

    # Filter data where mean_confidences < 0.2
    filter_indices = np.where(mean_confidences < 0.8 and variabilities_mean < 0.25)
    filtered_mean_confidences = mean_confidences[filter_indices]
    filtered_variabilities_mean = variabilities_mean[filter_indices]
    filtered_colors = plt.cm.coolwarm(correctness_means[filter_indices])
    filtered_markers = [markers[i] for i in filter_indices[0]]

    # Plotting for Mean Confidences vs. Variabilities with filtered data
    for x, y, m, c in zip(filtered_variabilities_mean, filtered_mean_confidences, filtered_markers, filtered_colors):
        axes[0].scatter(x, y, marker=m, color=c, alpha=0.5)
    axes[0].set_title('Filtered: Mean Confidences vs. Variabilities (<0.2)')
    axes[0].set_xlabel('Variability')
    axes[0].set_ylabel('Mean Confidence')
    axes[0].grid(True)

    # Apply similar filtering for geometric mean confidences if needed
    filter_indices_geom = np.where(geom_mean_confidences < 0.8 and variabilities_geom_mean < 0.04)
    filtered_geom_mean_confidences = geom_mean_confidences[filter_indices_geom]
    filtered_variabilities_geom_mean = variabilities_geom_mean[filter_indices_geom]
    filtered_colors_geom = plt.cm.coolwarm(correctness_means[filter_indices_geom])
    filtered_markers_geom = [markers[i] for i in filter_indices_geom[0]]

    # Plotting for Geometric Mean Confidences vs. Variabilities with filtered data
    for x, y, m, c in zip(filtered_variabilities_geom_mean, filtered_geom_mean_confidences, filtered_markers_geom, filtered_colors_geom):
        axes[1].scatter(x, y, marker=m, color=c, alpha=0.5)
    axes[1].set_title('Filtered: Geometric Mean Confidences vs. Variabilities (<0.2)')
    axes[1].set_xlabel('Variability')
    axes[1].set_ylabel('Geometric Mean Confidence')
    axes[1].grid(True)

    # Add colorbars and legends as needed
    plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), ax=axes[0]).set_label('Correctness')
    plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), ax=axes[1]).set_label('Correctness')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "filtered_epoch_plot_2.png"))


def process_dataset(data, instance_order, selected_instances, epochs, step_size):
    
    mean_confidences = np.array([])
    variabilities_mean = np.array([])
    mean_confidence = data['mean_confidence']
    print("Calculating mean confidence.")

    half = selected_instances[1] // 2
    mean_confidences = np.concatenate((mean_confidences, epoch_mean(instance_order, mean_confidence, epochs, [0, half], step_size)), axis=0)
    mean_confidences = np.concatenate((mean_confidences, epoch_mean(instance_order, mean_confidence, epochs, [half, selected_instances[1]], step_size)), axis=0)
    
    variabilities_mean = np.concatenate((variabilities_mean, epoch_variability(instance_order, mean_confidence, epochs, [0, half], step_size)), axis=0)
    variabilities_mean = np.concatenate((variabilities_mean, epoch_variability(instance_order, mean_confidence, epochs, [half, selected_instances[1]], step_size)), axis=0)

    del data['mean_confidence']
    del mean_confidence
    gc.collect()

    geom_mean_confidences = np.array([])
    variabilities_geom_mean = np.array([])

    geom_mean_confidence = data['geom_mean_confidence']
    print("Calculating geometric mean confidence.")
    geom_mean_confidences = np.concatenate((geom_mean_confidences, epoch_mean(instance_order, geom_mean_confidence, epochs, [0, half], step_size)), axis=0)
    variabilities_geom_mean = np.concatenate((variabilities_geom_mean, epoch_variability(instance_order, geom_mean_confidence, epochs, [0, half], step_size)), axis=0)

    geom_mean_confidences = np.concatenate((geom_mean_confidences, epoch_mean(instance_order, geom_mean_confidence, epochs, [half, selected_instances[1]], step_size)), axis=0)
    variabilities_geom_mean = np.concatenate((variabilities_geom_mean, epoch_variability(instance_order, geom_mean_confidence, epochs, [half, selected_instances[1]], step_size)), axis=0)

    del data['geom_mean_confidence']
    del geom_mean_confidence
    gc.collect()

    correctness = data['correctness']
    print("Calculating correctness.")
    correctness_means = epoch_mean(instance_order, correctness, epochs, selected_instances, step_size)

    print(np.mean(correctness_means))
    print(np.mean(mean_confidences))
    print(np.mean(geom_mean_confidences))
    print(np.mean(variabilities_mean))
    print(np.mean(variabilities_geom_mean))    
    print()

    return {
        "mean_confidences": mean_confidences,
        "geom_mean_confidences": geom_mean_confidences,
        "variabilities_mean": variabilities_mean,
        "variabilities_geom_mean": variabilities_geom_mean,
        "correctness_means": correctness_means
    }

def compute_partitions(args):

    data_train = load_train_data_from_hdf5(os.path.join(args.output_dir, "instances_masks.hdf5"))
    data_eval = load_eval_data_from_hdf5(os.path.join(args.output_dir, "dynamics_eval.hdf5"))

    instance_order = data_train['instance_order']
    print(len(instance_order))

    print(len(data_eval['mean_confidence']))
    epochs = []
    for i in range(len(data_eval['mean_confidence'][0])):
        if data_eval['mean_confidence'][len(data_eval['mean_confidence']) - 1][i] != 0:
            epochs.append(i)
            print(i)

    selected_instances = [0, len(data_eval['mean_confidence'])]

    results_eval = process_dataset(data_eval, instance_order, selected_instances, epochs, 1)

    mean_confidences_top_indices = select_top_indices(results_eval['mean_confidences'])

    variabilities_mean_top_indices = select_top_indices(results_eval['variabilities_mean'])

    mean_confidences_bottom_indices = select_bottom_indices(results_eval['mean_confidences'])

    mean_easy_instances = instance_order[mean_confidences_top_indices]
    mean_hard_instances = instance_order[mean_confidences_bottom_indices]
    mean_ambiguous_instances = instance_order[variabilities_mean_top_indices]

    os.makedirs(args.partition_data_path, exist_ok=False)

    initialize_hdf5_file(os.path.join(args.partition_data_path, "easy.hdf5"), mean_easy_instances)
    initialize_hdf5_file(os.path.join(args.partition_data_path, "hard.hdf5"), mean_hard_instances)
    initialize_hdf5_file(os.path.join(args.partition_data_path, "ambiguous.hdf5"), mean_ambiguous_instances)

def get_percent_indices(confidences, percent):
    num_elements = int(len(confidences) * (percent / 100))
    sorted_indices = np.argsort(confidences)
    return sorted_indices[:num_elements], sorted_indices[-num_elements:]

def compare_dynamics(args):
    model_path_first = "/project/data/models/BERT-6L-512H-10k"
    model_path_second = "/project/data/models/BERT-6L-512H-10k-20k"

    data_train_first = load_train_data_from_hdf5(os.path.join(model_path_first, "instances_masks.hdf5"))
    data_train_second = load_train_data_from_hdf5(os.path.join(model_path_second, "instances_masks.hdf5"))

    data_eval_first_seen = load_eval_data_from_hdf5(os.path.join(model_path_first, "dynamics_eval_42.hdf5"))
    data_eval_second_seen = load_eval_data_from_hdf5(os.path.join(model_path_second, "dynamics_eval_42.hdf5"))
    
    data_eval_first_unseen = load_eval_data_from_hdf5(os.path.join(model_path_first, "dynamics_eval_unseen_42.hdf5"))
    data_eval_second_unseen = load_eval_data_from_hdf5(os.path.join(model_path_second, "dynamics_eval_unseen_42.hdf5"))
    
    instance_order_first = data_train_first['instance_order']
    instance_order_second = data_train_second['instance_order']

    mean_confidence_first_seen = data_eval_first_seen['mean_confidence'][-1]
    mean_confidence_first_unseen = data_eval_first_unseen['mean_confidence'][-1]
 
    mean_confidence_second_seen = data_eval_second_seen['mean_confidence'][-1]
    mean_confidence_second_unseen = data_eval_second_unseen['mean_confidence'][-1]

    # 1. Comparing instance orders
    shared_instances = np.intersect1d(instance_order_first, instance_order_second)
    print(f"Shared instances between Model 1 and Model 2: {len(shared_instances)}")

    # 2. Confidence scores comparison
    print("Mean confidence on seen data (Model 1):", np.mean(mean_confidence_first_seen))
    print("Mean confidence on unseen data (Model 1):", np.mean(mean_confidence_first_unseen))
    print("Mean confidence on seen data (Model 2):", np.mean(mean_confidence_second_seen))
    print("Mean confidence on unseen data (Model 2):", np.mean(mean_confidence_second_unseen))

    # Analyze top and bottom 10, 20, 30 percent
    for frac in [10, 20, 30, 40, 50]:

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
        print(f"Intersection of top {frac}% between Model 1 Seen and Model 2 Seen: {len(top_instances_first_seen & top_instances_second_unseen)} / {len(top_instances_first_seen)}")
        print(f"Intersection of bottom {frac}% between Model 1 Seen and Model 2 Seen: {len(bottom_instances_first_seen & bottom_instances_second_unseen)} / {len(top_instances_first_seen)}")
        
        print(f"Intersection of top {frac}% between Model 1 Unseen and Model 2 Unseen: {len(top_instances_first_unseen & top_instances_second_seen)} / {len(top_instances_first_seen)}")
        print(f"Intersection of bottom {frac}% between Model 1 Unseen and Model 2 Unseen: {len(bottom_instances_first_unseen & bottom_instances_second_seen)} / {len(top_instances_first_seen)}")
        print()

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

    mean_confidences_top_indices = select_top_indices(confidence)
    mean_entropy_top_indices = select_top_indices(entropy)

    mean_confidences_bottom_indices = select_bottom_indices(confidence)
    mean_entropy_bottom_indices = select_bottom_indices(entropy)

    mean_easy_instances = instance_order[mean_confidences_top_indices]
    seed_mean_easy_instances = instance_order[mean_entropy_top_indices]

    mean_hard_instances = instance_order[mean_confidences_bottom_indices]
    mean_entropy_hard_instances = instance_order[mean_entropy_bottom_indices]

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

    
def sample_evenly(data, target_length):
    data = np.array(data)
    indices = np.linspace(0, len(data) - 1, target_length, dtype=int)
    return data[indices]

def plots(args):
    # Load data from log files
    data_easy = extract_data_from_log("/home/robert/Documents/ETT/data/logs/results/third_easy.txt")
    data_ambig = extract_data_from_log("/home/robert/Documents/ETT/data/logs/results/third_amb.txt")
    data_hard = extract_data_from_log("/home/robert/Documents/ETT/data/logs/results/third_hard.txt")

    # Extract training loss and steps from each dataset
    #train_loss_easy, steps_easy = sample_evenly(data_easy["train_loss"], 152), sample_evenly(data_easy["train_step"], 152)
    #train_loss_ambig, steps_ambig = sample_evenly(data_ambig["train_loss"], 152), sample_evenly(data_ambig["train_step"], 152)
    #train_loss_hard, steps_hard = sample_evenly(data_hard["train_loss"], 152), sample_evenly(data_hard["train_step"], 152)

    train_loss_easy, steps_easy = data_easy["train_loss"], data_easy["train_step"]
    train_loss_ambig, steps_ambig = data_ambig["train_loss"], data_ambig["train_step"]
    train_loss_hard, steps_hard = data_hard["train_loss"], data_hard["train_step"]

    # Extract evaluation perplexity and steps from each dataset
    #eval_ppl_easy = sample_evenly(data_easy["eval_ppl"], 152)
    #eval_ppl_ambig = sample_evenly(data_ambig["eval_ppl"], 152)
    #eval_ppl_hard = sample_evenly(data_hard["eval_ppl"], 152)
    
    eval_ppl_easy = data_easy["eval_ppl"]
    eval_ppl_ambig = data_ambig["eval_ppl"]
    eval_ppl_hard = data_hard["eval_ppl"]

    print(len(eval_ppl_easy))
    print(len(eval_ppl_hard))
    print(len(eval_ppl_ambig))
    # Create a plot for training loss
    plt.figure(figsize=(10, 5))
    plt.plot(steps_easy, train_loss_easy, label="Easy")
    plt.plot(steps_ambig, train_loss_ambig, label="Ambiguous")
    plt.plot(steps_hard, train_loss_hard, label="Hard")
    plt.title("Training Loss Comparison")
    plt.xlabel("Training Steps")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.savefig("/home/robert/Documents/ETT/data/logs/results/train_loss_third")

    # Create a plot for evaluation perplexity
    plt.figure(figsize=(10, 5))
    plt.plot(steps_easy, np.log(eval_ppl_easy), label="Easy")
    plt.plot(steps_ambig, np.log(eval_ppl_ambig), label="Ambiguous")
    plt.plot(steps_hard, np.log(eval_ppl_hard), label="Hard")
    plt.title("Evaluation Perplexity Comparison")
    plt.xlabel("Training Steps")
    plt.ylabel("Evaluation Perplexity")
    plt.legend()
    plt.savefig("/home/robert/Documents/ETT/data/logs/results/test_loss_third")

    # Determine the number of steps to zoom in on
    zoom_steps = 100  # Change this number to adjust the zoom level

    # Create a plot for evaluation perplexity using a logarithmic scale, focusing on the last few steps
    plt.figure(figsize=(10, 5))
    plt.plot(steps_easy[-zoom_steps:], np.log(eval_ppl_easy[-zoom_steps:]), label="Easy", marker='o')
    plt.plot(steps_ambig[-zoom_steps:], np.log(eval_ppl_ambig[-zoom_steps:]), label="Ambiguous", marker='o')
    plt.plot(steps_hard[-zoom_steps:], np.log(eval_ppl_hard[-zoom_steps:]), label="Hard", marker='o')
    plt.title("Zoomed Evaluation Perplexity Comparison (Log Scale)")
    plt.xlabel("Training Steps")
    plt.ylabel("Log of Evaluation Perplexity")
    plt.legend()
    plt.savefig("/home/robert/Documents/ETT/data/logs/results/test_loss_zoomed_third")  # Save the plot to a file

        # Determine the number of steps to zoom in on
    zoom_steps = 60  # Change this number to adjust the zoom level

    # Create a plot for evaluation perplexity using a logarithmic scale, focusing on the last few steps
    plt.figure(figsize=(10, 5))
    plt.plot(steps_easy[-zoom_steps*20:-zoom_steps*19], np.log(eval_ppl_easy[-zoom_steps*20:-zoom_steps*19]), label="Easy", marker='o')
    plt.plot(steps_ambig[-zoom_steps*20:-zoom_steps*19], np.log(eval_ppl_ambig[-zoom_steps*20:-zoom_steps*19]), label="Ambiguous", marker='o')
    plt.plot(steps_hard[-zoom_steps*20:-zoom_steps*19], np.log(eval_ppl_hard[-zoom_steps*20:-zoom_steps*19]), label="Hard", marker='o')
    plt.title("Zoomed Evaluation Perplexity Comparison (Log Scale)")
    plt.xlabel("Training Steps")
    plt.ylabel("Log of Evaluation Perplexity")
    plt.legend()
    plt.savefig("/home/robert/Documents/ETT/data/logs/results/test_loss_zoomed_third2")  # Save the plot to a file

def check_duplicates(list):
    int_array = np.array(list)
    print(len(int_array))
    unique_elements, counts = np.unique(int_array, return_counts=True)
    print(len(unique_elements))
    return np.any(counts > 1)

def compare_partitions(partition1, partition2, partition3):
    set1 = set(partition1)
    set2 = set(partition2)
    set3 = set(partition3)

    # Unique elements in each partition
    unique_to_p1 = set1 - (set2.union(set3))
    unique_to_p2 = set2 - (set1.union(set3))
    unique_to_p3 = set3 - (set1.union(set2))

    # Common elements among any two
    common_p1_p2 = set1.intersection(set2) - set3
    common_p2_p3 = set2.intersection(set3) - set1
    common_p1_p3 = set1.intersection(set3) - set2

    # Common elements among all three
    common_all = set1.intersection(set2).intersection(set3)

    return {
        "unique_to_p1": unique_to_p1,
        "unique_to_p2": unique_to_p2,
        "unique_to_p3": unique_to_p3,
        "common_p1_p2": common_p1_p2,
        "common_p2_p3": common_p2_p3,
        "common_p1_p3": common_p1_p3,
        "common_all": common_all
    }



def main():
    parser = process_args()
    args = parser.parse_args()

    # Load data
    amb_instances = load_train_data_from_hdf5("/home/robert/Documents/ETT/data/partitions/12L-768H-full-third/ambiguous.hdf5")['instance_order']
    easy_instances = load_train_data_from_hdf5("/home/robert/Documents/ETT/data/partitions/12L-768H-full-third/easy.hdf5")['instance_order']
    hard_instances = load_train_data_from_hdf5("/home/robert/Documents/ETT/data/partitions/12L-768H-full-third/hard.hdf5")['instance_order']

    # Compare partitions
    comparison_results = compare_partitions(amb_instances, easy_instances, hard_instances)

    # Print results
    for key, value in comparison_results.items():
        print(f"{key}: {len(value)} elements")
    #data = load_train_data_from_hdf5(os.path.join(args.output_dir, "instances_masks.hdf5"))
    #data_comp = load_train_data_from_hdf5(os.path.join(args.dynamics_path, "instances_masks.hdf5"))
    #plots(args)
    if args.compute_dynamics:  
        compute_partitions(args)


if __name__ == "__main__":
    main()
