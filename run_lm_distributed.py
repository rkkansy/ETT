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
import threading
import h5py
from torch.cuda.amp import autocast, GradScaler


import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.optim.lr_scheduler import OneCycleLR

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

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, SimpleBertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, SimpleRobertaForMaskedLM, RobertaTokenizer),
}


class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file_bc if args.add_bc else args.train_data_file
    if args.col_data:
        return CoLDataset(file_path, args.tokenizer_name, tokenizer, args.block_size,
                          split_sent=args.split_sent,
                          verbose=True)
    elif args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args, get_mask=False):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    
    labels = inputs.clone()
    operation_mask = torch.zeros(labels.shape, dtype=torch.int8)

    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    rand_words = random_words[indices_random]

    inputs[indices_random] = rand_words
    
    if get_mask:
        operation_mask[indices_replaced] = 1  # Masked tokens marked as 1
        operation_mask[indices_random] = 2  # Randomized tokens marked as 2
        indices_kept = masked_indices & ~indices_replaced & ~indices_random
        operation_mask[indices_kept] = 3

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels, operation_mask

def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    set_seed(args)  # Added here for reproducibility
    info = True

    """ Train the model """

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    tb_writer = SummaryWriter(args.output_dir + '/runs/' + current_time)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    t_total = args.max_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    if args.scheduler_type == 'one_cycle':
        args.learning_rate = 4e-5
        
    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(0.9, 0.98),
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    if args.warmup_ratio > 0.:
        assert args.warmup_steps == 0
        args.warmup_steps = int(t_total * args.warmup_ratio)

    print("Optimized with lr %f, steps %d, warmup steps %d, and use beta, epsilon %0.8f." % (
        args.learning_rate, t_total, args.warmup_steps, optimizer.defaults['eps']), 
        optimizer.defaults['betas'])
    
    if args.scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    elif args.scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total,
            num_cycles=args.scheduler_cosine_cycles
        )
    elif args.scheduler_type == 'poly':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total,
            power=args.scheduler_poly_power
        )
    elif args.scheduler_type == 'one_cycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.001,
            total_steps=t_total,
            pct_start=0.5,
            anneal_strategy='linear',
            cycle_momentum=False, 
            div_factor=25.0,    
            final_div_factor=10000.0
        )
    else:
        raise ValueError(f"Unknow lr scheduler: {args.scheduler_type}")

    scaler = GradScaler()

    # Check if saved optimizer or scheduler states exist
    if (args.model_name_or_path and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) 
                                and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
                                and os.path.isfile(os.path.join(args.model_name_or_path, "scaler.pt"))):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"), map_location=torch.device('cpu')))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"), map_location=torch.device('cpu')))
        scaler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scaler.pt"), map_location=torch.device('cpu')))
    
    # Train!
    logger.info("***** Running training *****")
    if info:
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        
    global_step = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_name = os.path.basename(args.model_name_or_path)
            global_step = int(checkpoint_name.split("-")[-1])
            logger.info("  Continuing training from iter %d" % (global_step))

        except ValueError:
            logger.info("  Do not load model from %s, restart training" % args.model_name_or_path)

    model.zero_grad()

    batch_size = args.train_batch_size * args.gradient_accumulation_steps
    epoch_size = args.max_steps * batch_size

    # IMPORTANT: save the initialization
    if global_step == 0:

        checkpoint_name = f"checkpoint-{global_step:08d}"
        ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        save_model(args, ckpt_dir, checkpoint_name, model, tokenizer, optimizer, scheduler, scaler)

        if args.data_partition in ['none', 'rand']:
            instance_list = list(range(0, len(train_dataset)))
        else:
            instance_list  = load_train_data_from_hdf5(os.path.join(args.partition_data_path, f"{args.data_partition}.hdf5"))['instance_order']

        random.shuffle(instance_list)
        if epoch_size < len(instance_list):
            instance_list = instance_list[:epoch_size]
    
        initialize_hdf5_file(os.path.join(args.output_dir, "instances_masks.hdf5"), instance_list)

    else:
        instance_list = load_train_data_from_hdf5(os.path.join(args.output_dir, "instances_masks.hdf5"))['instance_order']

    epochs_trained = global_step // (len(instance_list) // args.train_batch_size // args.gradient_accumulation_steps)
    steps_per_epoch = len(instance_list) // (args.train_batch_size * args.gradient_accumulation_steps)
    start_index = (global_step % steps_per_epoch) * args.gradient_accumulation_steps

    train_sampler = CustomSampler(instance_list)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, shuffle=False, num_workers=0,
                                  batch_size=args.train_batch_size, collate_fn=collate, pin_memory=True
    )
    while True:
        epoch_iterator = tqdm(train_dataloader, 
                              desc=f"Steps trained: {global_step:06d}", 
                              ncols=100)
        tr_loss, tr_lm_loss = 0.0, 0.0
        t_start = time.time()
        model.zero_grad()       # Support of accumulating gradients

        for step, batch in enumerate(epoch_iterator):

            if step < start_index:
                continue

            inputs, labels, _ = mask_tokens(batch, tokenizer, args)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            # If some of the input is padded, then the attention mask is needed
            attention_mask = (inputs != tokenizer.pad_token_id)         # word_tokens --> 1, pad_token --> 0
            if attention_mask.all():
                attention_mask = None

            model.train()
            with autocast():
                outputs = model(inputs,
                    attention_mask=attention_mask,
                    masked_lm_labels=labels,
                    current_step=global_step)
            
                loss = outputs['loss']  # model outputs are always tuple in transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            tr_loss += loss.item()
            tr_lm_loss += outputs['lm_loss'].item()

            if (step + 1) % args.gradient_accumulation_steps == 0:

                if args.max_grad_norm > 0.:
                    scaler.unscale_(optimizer) 
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    tb_writer.add_scalar("grad_norm", total_norm, global_step)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step() 
                optimizer.zero_grad()
                global_step += 1
                
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:

                    t_elapse = time.time() - t_start

                    # Log metrics
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss_scale", scaler.get_scale(), global_step)

                    train_loss = tr_loss / args.logging_steps
                    train_ppl = torch.exp(torch.tensor(tr_lm_loss / (args.logging_steps * args.gradient_accumulation_steps))).item()
                    tb_writer.add_scalar("loss", train_loss, global_step)
                    tb_writer.add_scalar("train_ppl", train_ppl, global_step)
                    tr_loss = tr_lm_loss = 0.

                    # also evaluate on valid set for ppl
                    logger.info(" Evaluation Results of step %d: " % global_step)
                    results = evaluate(args, model, tokenizer)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        logger.info("\t %s: %0.4f" % (key, value))

                    output_log_file = os.path.join(args.output_dir, "train_log.txt")
                    with open(output_log_file, 'a') as f:
                        eval_ppl = results['perplexity']
                        print(f"train_step={global_step}, train_time={t_elapse}, lr={scheduler.get_last_lr()[0]}, train_loss={train_loss}, "
                            f"scale={scaler.get_scale()}, grad_norm={total_norm}, train_ppl={train_ppl}, eval_ppl={eval_ppl}", file=f)
                
                t_start = time.time()
                
                if args.ckpt_steps > 0 and global_step % args.ckpt_steps == 0:
                    checkpoint_name = f"checkpoint-{global_step:08d}"
                    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
                    os.makedirs(ckpt_dir, exist_ok=True)
                    save_model(args, ckpt_dir, checkpoint_name, model, tokenizer, optimizer, scheduler, scaler)
            
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if args.max_steps > 0 and global_step >= args.max_steps:
            epoch_iterator.close()
            break
        start_index = 0
        epochs_trained += 1
        args.seed = epochs_trained

    tb_writer.close()

def evaluate_train(args, train_dataset, instance_list, eval_run, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix=""):
    set_seed(args) 
    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    train_sampler = CustomSampler(instance_list)

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, shuffle=False, num_workers=2,
        batch_size=args.eval_batch_size, collate_fn=collate, pin_memory=True
    )

    correctness = np.zeros((args.logging_steps * args.eval_batch_size), dtype=np.float32)
    mean_confidence = np.zeros((args.logging_steps * args.eval_batch_size), dtype=np.float32)
    geom_mean_confidence = np.zeros((args.logging_steps * args.eval_batch_size), dtype=np.float32)

    save_thread = None
    for step, batch in enumerate(tqdm(train_dataloader, desc="Evaluating", ncols=100)):
    
        inputs, labels, _ = mask_tokens(batch, tokenizer, args)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        # If some of the input is padded, then the attention mask is needed
        attention_mask = (inputs != tokenizer.pad_token_id)  # word_tokens --> 1, pad_token --> 0
        if attention_mask.all():
            attention_mask = None

        with torch.no_grad():
            with autocast():
                outputs = model(inputs, attention_mask=attention_mask, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
    
            logits = outputs['logits']
            
            insert_idx = (step % args.logging_steps) * args.eval_batch_size 
            insert_idx_batch = insert_idx + args.eval_batch_size
            softmax_logits = torch.softmax(logits, dim=-1)  
            valid_labels_mask = labels != -100
            valid_labels_mask_float = valid_labels_mask.float()

            # Calculate correctness per instance
            predicted_token_ids = torch.argmax(logits, dim=-1)
            correct_predictions_mask = (predicted_token_ids == labels) & valid_labels_mask
            correct_predictions_per_instance = correct_predictions_mask.sum(dim=1).float()
            total_considered_per_instance = valid_labels_mask_float.sum(dim=1)

            # Avoid division by 0 in case no words are masked
            total_considered_per_instance = total_considered_per_instance.masked_fill_(total_considered_per_instance == 0, 1.)
            correctness_per_instance = correct_predictions_per_instance / total_considered_per_instance
            
            # Gather correct token prediction probabilities
            labels_adjusted = labels.clone()
            labels_adjusted[~valid_labels_mask] = 0 
            correct_token_probs = torch.gather(softmax_logits, dim=-1, index=labels_adjusted.unsqueeze(-1))

            # Calculate mean confidence
            sum_probs = torch.sum(correct_token_probs.squeeze(-1) * valid_labels_mask, dim=1)
            mean_probs = sum_probs / total_considered_per_instance

            # Calculate geometric mean in log-space to mitigate underflow
            log_probs = torch.log(correct_token_probs.squeeze(-1) + 1e-7)
            sum_log_probs = torch.sum(log_probs * valid_labels_mask, dim=1)
            geom_mean_log = sum_log_probs / total_considered_per_instance
            geom_mean_probs = torch.exp(geom_mean_log)               

            correctness[insert_idx : insert_idx_batch] = correctness_per_instance.detach().cpu().numpy()
            mean_confidence[insert_idx : insert_idx_batch] = mean_probs.detach().cpu().numpy()
            geom_mean_confidence[insert_idx : insert_idx_batch] = geom_mean_probs.detach().cpu().numpy()

        if (step + 1) % args.logging_steps == 0 and step > 0:
            
            if save_thread is not None:
                save_thread.join()

            offset = (step + 1 - args.logging_steps) * args.eval_batch_size
            
            correctness_copy = correctness.copy()
            mean_confidence_copy = mean_confidence.copy()
            geom_mean_confidence_copy = geom_mean_confidence.copy()
            save_thread = threading.Thread(target=async_add_probs_batch, 
                                           args=(args, args.output_dir, offset, correctness_copy, mean_confidence_copy, geom_mean_confidence_copy, eval_run))
            save_thread.start()
        
        if step + 1 >= args.max_steps:
            break
    
    if save_thread is not None:
        save_thread.join()

def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating", ncols=100):
        inputs, labels, _ = mask_tokens(batch, tokenizer, args)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        # If some of the input is padded, then the attention mask is needed
        attention_mask = (inputs != tokenizer.pad_token_id)  # word_tokens --> 1, pad_token --> 0
        if attention_mask.all():
            attention_mask = None

        with torch.no_grad():
            outputs = model(inputs, attention_mask=attention_mask, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs['lm_loss']
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss)).item()

    result = {"perplexity": perplexity}

    return result

def compute_dynamics(args, train_dataset, tokenizer):

    # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO
        )
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        ckpt_dir = os.path.join(args.output_dir, 'checkpoints')

        checkpoint_names = []
        model_names = []
        if os.path.isdir(ckpt_dir):
            checkpoint_names = [fn for fn in os.listdir(ckpt_dir) if fn.startswith('checkpoint-')]
        if len(checkpoint_names) > 0:
            checkpoint_names = sorted(checkpoint_names, key=lambda p: int(p.split('-')[-1]))
            for i, ckpt_name in enumerate(checkpoint_names):
                model_names.append(os.path.join(ckpt_dir, ckpt_name))
        else:
            logger.warning('No checkpoint detected: %s', ckpt_dir)
            return -1

        assert args.block_size <= tokenizer.model_max_length

        # Get Config
        if args.config_name:
            config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
        elif args.model_name_or_path:
            config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            raise ValueError(
                "Why do you want the default config?? Please use --config_name or --model_name_or_path"
            )
        
        if args.dynamics_path == "none":
            args.dynamics_path = args.output_dir

        data = load_train_data_from_hdf5(os.path.join(args.dynamics_path, "instances_masks.hdf5"))
        instance_list = data['instance_order']

        args.max_steps = len(instance_list) // args.eval_batch_size
        args.logging_steps *= args.gradient_accumulation_steps
        args.output_dir = os.path.join(args.output_dir, "dynamics_eval_rdm_masks.hdf5") if args.random_masks else os.path.join(args.output_dir, "dynamics_eval.hdf5") 

        if not os.path.isfile(args.output_dir):
            initialize_hdf5_file_eval(args.output_dir, args.max_steps * args.eval_batch_size,  len(model_names))
        
        eval_thread = None
        for i in args.dynamics_ckpts_list:

                if eval_thread is not None:
                    eval_thread.join()

                args.model_name_or_path = model_names[i]
                print(f"Evaluating {args.model_name_or_path} for {args.max_steps} steps.")

                model = model_class.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    cache_dir=args.cache_dir,
                    args=args
                )
                model.to(args.device)
                if args.random_masks:
                    args.seed = 42
                
                #eval_thread = threading.Thread(target=evaluate_train, args=(args, train_dataset, instance_list, i, model, tokenizer))
                #eval_thread.start()
                evaluate_train(args, train_dataset, instance_list, i, model, tokenizer)

        if eval_thread is not None:
            eval_thread.join()


def save_model(args, ckpt_dir, name, model, tokenizer, optimizer, scheduler, scaler):
    # Save model checkpoint
    output_dir = os.path.join(ckpt_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    torch.save(scaler.state_dict(), os.path.join(output_dir, "scaler.pt"))

    logger.info("Saving optimizer and scheduler states to %s", output_dir)

def get_model_tokenizer(args):
        # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    if args.should_continue:
        ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
        checkpoint_names = []
        if os.path.isdir(ckpt_dir):
            checkpoint_names = [fn for fn in os.listdir(ckpt_dir) if fn.startswith('checkpoint-')]
        if len(checkpoint_names) > 0:
            checkpoint_names = sorted(checkpoint_names, key=lambda p: int(p.split('-')[-1]))
            args.model_name_or_path = os.path.join(ckpt_dir, checkpoint_names[-1])
        else:
            logger.warning('No checkpoint detected: %s', ckpt_dir)
            args.model_name_or_path = None

    # Set seed
    set_seed(args)
    # Load pretrained model and token
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Get Config
    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "Why do you want the default config?? Please use --config_name or --model_name_or_path"
        )

    # Get Tokenizer
    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
        # BERT always needs lower cased tokens.
        if 'uncased' in args.model_type:
            assert tokenizer.init_kwargs.get("do_lower_case", False)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, "
            "but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    assert args.block_size <= tokenizer.model_max_length

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            args=args
        )
    elif args.source_model_path or args.pretrained_ligo_path:
        logger.info("Growing [%s] new model from: %s", args.grow_scheme, args.source_model_path)

        model = model_class(config=config, args=args)

        if args.grow_scheme == 'none':
            logger.info("No initialization scheme applied. Training new model with random initialization ...")
        elif args.grow_scheme == 'ligo':
            ckpt_dir = os.path.join(args.pretrained_ligo_path, 'checkpoints')
            checkpoint_names = [fn for fn in os.listdir(ckpt_dir) if fn.startswith('checkpoint-')]
            checkpoint_names = sorted(checkpoint_names, key=lambda p: int(p.split('-')[-1]))
            args.pretrained_ligo_path = os.path.join(ckpt_dir, checkpoint_names[-1])

            args.fuse_init_scheme_depth = args.fuse_init_scheme_width = args.fuse_init_scheme[0]
            if len(args.fuse_init_scheme) >= 2:
                args.fuse_init_scheme_width = args.fuse_init_scheme[1]
            args.fuse_init_noise_depth = args.fuse_init_noise_width = args.fuse_init_noise[0]
            if len(args.fuse_init_noise) >= 2:
                args.fuse_init_noise_width = args.fuse_init_noise[1]

            model = initialize_model_with_ligo(model, args)
        else:
            raise NotImplementedError(f'Grow method [{args.grow_scheme}] not implemented yet!')

    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config, args=args)

    return model, tokenizer

def main():
    parser = process_args()
    args = parser.parse_args()
    set_seed(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_tokenizer(args)
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

    train_thread = None
    # Training
    if args.do_train:
        model.to(args.device)
        train(args, train_dataset, model, tokenizer)
        #train_thread = threading.Thread(target=train, args=(args, train_dataset, model, tokenizer))
        #train_thread.start()

    if args.compute_dynamics:
        args.dynamics_ckpts_list = [5, 10, 15, 19]
        compute_dynamics(args, train_dataset, tokenizer)

    if train_thread is not None:
        train_thread.join()

if __name__ == "__main__":
    main()
