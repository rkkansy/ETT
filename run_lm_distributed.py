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
from model import SimpleBertForMaskedLM

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from ligo import initialize_model_with_ligo

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, SimpleBertForMaskedLM, BertTokenizer),
}

def load_and_cache_examples(args, tokenizer, evaluate=False):
    if evaluate:
        file_path = args.eval_data_file
        mask_path = None
    else: 
        file_path = args.train_data_file_bc if args.add_bc else args.train_data_file
        mask_path=None if args.generate_masks else args.mask_path

    return CoLDataset(file_path, 
                        args.tokenizer_name, 
                        tokenizer, 
                        mask_path=mask_path,
                        block_size=args.block_size,
                        split_sent=args.split_sent,
                        verbose=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    labels = inputs.clone()
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
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def mask_inputs(inputs: torch.Tensor, operation_mask: torch.Tensor, tokenizer: PreTrainedTokenizer):

    # Non masked tokens should be ignored (mask=0)
    labels = inputs.clone()
    labels[operation_mask == 0] = -100
    # Indices larger than one are either unchanged or randomized tokens
    random_unchanged_indices = operation_mask > 0
    inputs[random_unchanged_indices] = operation_mask[random_unchanged_indices]
    # -1 indicates a mask
    inputs[operation_mask == -1] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token) 

    return inputs, labels

def generate_masks(args, train_dataset, tokenizer: PreTrainedTokenizer):
    set_seed(args.mask_set)  # Added here for reproducibility

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    global_step = 0
    instances = list(range(len(train_dataset)))

    with h5py.File(args.mask_path, 'a') as f:
        f.create_dataset("masks", shape=(len(instances), 128), dtype=np.int16)

    train_sampler = CustomSampler(instances)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, shuffle=False, num_workers=0,
                                  batch_size=args.train_batch_size, collate_fn=collate, pin_memory=True
    )
    masks = np.zeros((args.logging_steps * args.train_batch_size, 128), dtype=np.int16)

    epoch_iterator = tqdm(train_dataloader, 
                            desc=f"Masks generated: {global_step:06d}", 
                            ncols=100)
    save_thread = None

    for step, batch in enumerate(epoch_iterator):
        
        if tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        operation_mask = torch.zeros(batch.shape, dtype=torch.long)
        probability_matrix = torch.full(batch.shape, args.mlm_probability)
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in batch.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if tokenizer._pad_token is not None:
            padding_mask = batch.eq(tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(batch.shape, 0.8)).bool() & masked_indices
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(batch.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # 10 % of the time, we dont replace the token
        random_words = torch.randint(len(tokenizer), batch.shape, dtype=torch.long)
        indices_kept = masked_indices & ~indices_replaced & ~indices_random

        operation_mask[indices_replaced] = -1
        operation_mask[indices_random] = random_words[indices_random]  # Store the actual random token IDs
        operation_mask[indices_kept] = batch[indices_kept] # Store the original token IDs

        insert_idx = (step % args.logging_steps) * args.train_batch_size 

        masks[insert_idx : insert_idx + args.train_batch_size] = operation_mask

        # Implement thread pool to save data faster
        if (step + 1) % args.logging_steps == 0:
            if save_thread is not None:
                save_thread.join()
            offset = (step + 1 - args.logging_steps) * args.train_batch_size
            masks_copy = masks.copy()
            save_thread = threading.Thread(target=async_save_masks, args=(args.mask_path, offset, masks_copy))
            save_thread.start()

    save_thread.join()

def compare_inputs_and_labels(inputs1, labels1, inputs2, labels2):
    # Check if both inputs sets are identical
    assert torch.equal(inputs1, inputs2), "Inputs do not match."

    # Check if both labels sets are identical
    assert torch.equal(labels1, labels2), "Labels do not match."

def get_training_components(args, model):
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
        args.warmup_steps = int(args.max_steps * args.warmup_ratio)

    print("Optimized with lr %f, steps %d, warmup steps %d, and use beta, epsilon %0.8f." % (
        args.learning_rate, args.max_steps, args.warmup_steps, optimizer.defaults['eps']), 
        optimizer.defaults['betas'])
    
    if args.scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
        )
    elif args.scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps,
            num_cycles=args.scheduler_cosine_cycles
        )
    elif args.scheduler_type == 'poly':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps,
            power=args.scheduler_poly_power
        )
    elif args.scheduler_type == 'one_cycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.001,
            total_steps=args.max_steps,
            pct_start=0.5,
            anneal_strategy='linear',
            cycle_momentum=False, 
            div_factor=25.0,    
            final_div_factor=10000.0
        )
    else:
        raise ValueError(f"Unknow lr scheduler: {args.scheduler_type}")

    scaler = GradScaler(init_scale=args.init_grad_scale)

        # Check if saved optimizer or scheduler states exist
    if (args.model_name_or_path and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) 
                                and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
                                and os.path.isfile(os.path.join(args.model_name_or_path, "scaler.pt"))):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"), map_location=torch.device('cpu')))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"), map_location=torch.device('cpu')))
        scaler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scaler.pt"), map_location=torch.device('cpu')))

    return optimizer, scheduler, scaler

def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    set_seed(args.seed)  # Added here for reproducibility
    info = True

    """ Train the model """
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    tb_writer = SummaryWriter(args.output_dir + '/runs/' + current_time)

    def collate(examples: List[Tuple[torch.Tensor, torch.Tensor]]):
        tokens, masks = zip(*examples)  
        
        if tokenizer._pad_token is None:
            padded_tokens = pad_sequence(tokens, batch_first=True)
        else:
            padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
        
        padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)
        return padded_tokens, padded_masks

    optimizer, scheduler, scaler = get_training_components(args, model)
    
    # Train!
    logger.info("***** Running training *****")
    if info:
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", args.max_steps)
        
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

    if global_step >= args.max_steps:
        logger.warning("Max steps lower than trained steps. Aborting")
        return
    
    model.zero_grad()
    
    batch_size = args.train_batch_size * args.gradient_accumulation_steps
    epoch_size = args.max_steps * batch_size

    if global_step == 0:

        checkpoint_name = f"checkpoint-{global_step:08d}"
        ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        save_model(args, ckpt_dir, checkpoint_name, model, tokenizer, optimizer, scheduler, scaler)
        if args.instance_data_path:
            load_path = os.path.join(args.instance_data_path, f"{args.data_partition}.hdf5")
            with h5py.File(load_path, 'r') as f:
                instances = f['instance_order'][:]
            logger.info(f"Loading instance indices from: {load_path}")

            with h5py.File(os.path.join(args.output_dir, "instance_data.hdf5"), 'a') as f:
                f.create_dataset(f"instance_order", data=instances, dtype=np.int32)
        else:
            instances = list(range(len(train_dataset)))[:args.train_batch_size * 100]
            if len(instances) < epoch_size:
                random.shuffle(instances)
                instances = instances[:epoch_size]
            else:
                random.shuffle(instances)

            with h5py.File(os.path.join(args.output_dir, "instance_data.hdf5"), 'a') as f:
                f.create_dataset(f"instance_order", data=instances, dtype=np.int32)
    else:
        if args.instance_data_path:
            load_path = os.path.join(args.instance_data_path, f"{args.data_partition}.hdf5")
        else:
            load_path = os.path.join(args.output_dir, "instance_data.hdf5")

        with h5py.File(load_path, 'r') as f:
            instances = f['instance_order'][:]
        logger.info(f"Loading instance indices from: {load_path}")

    instance_amount = len(instances) 

    steps_per_epoch = instance_amount // (args.train_batch_size * args.gradient_accumulation_steps)
    start_index = (global_step % steps_per_epoch) * args.gradient_accumulation_steps
    instances = instances[start_index * args.train_batch_size:]

    epochs_trained = global_step // (instance_amount // args.train_batch_size // args.gradient_accumulation_steps)
    
    if args.shuffle:
        random.shuffle(instances)

    train_sampler = CustomSampler(instances)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, shuffle=False, num_workers=0,
                                  batch_size=args.train_batch_size, collate_fn=collate, pin_memory=True
    )
    if args.get_dynamics:
        with h5py.File(args.dynamics_path, 'a') as f:
            f.create_dataset("confidence_ckpt_0", shape=(epoch_size), dtype=np.float32)
            f.create_dataset("entropy_ckpt_0", shape=(epoch_size), dtype=np.float32)
            f.create_dataset("correctness_ckpt_0", shape=(epoch_size), dtype=np.float32)

        correctness = np.zeros((args.logging_steps * args.train_batch_size), dtype=np.float32)
        confidence = np.zeros((args.logging_steps * args.train_batch_size), dtype=np.float32)
        entropy = np.zeros((args.logging_steps * args.train_batch_size), dtype=np.float32)

    save_thread = None

    while True:
        epoch_iterator = tqdm(train_dataloader, 
                              desc=f"Steps trained: {global_step:06d}", 
                              ncols=100)
        tr_loss, tr_lm_loss = 0.0, 0.0
        t_start = time.time()
        model.zero_grad()       # Support of accumulating gradients

        logger.info(f"Epochs trained: {epochs_trained} | Seed: {args.seed}")

        for step, batch in enumerate(epoch_iterator):
            inputs, labels = mask_inputs(batch[0], batch[1], tokenizer)
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

                if args.get_dynamics:
                    with torch.no_grad():
                        logits = outputs['logits']
                        softmax_logits = torch.softmax(logits, dim=-1)  

                        insert_idx = (step % args.logging_steps) * args.train_batch_size 
                        insert_idx_batch = insert_idx + args.train_batch_size

                        valid_labels_mask = labels != -100
                        valid_labels_mask_float = valid_labels_mask.float()
                        amount_masked = torch.sum(valid_labels_mask_float, dim=1)
                        # Avoid division by 0 in case no words are masked
                        amount_masked = amount_masked.masked_fill_(amount_masked == 0, 1.)

                        # Calculate correctness per instance
                        predicted_token_ids = torch.argmax(logits, dim=-1)
                        correct_predictions_mask = (predicted_token_ids == labels) & valid_labels_mask
                        correct_predictions = torch.sum(correct_predictions_mask.float(), dim=1)
                        correctness_batch = correct_predictions / amount_masked
                        correctness[insert_idx : insert_idx_batch] = correctness_batch.detach().cpu().numpy()

                        # Gather correct token prediction probabilities
                        labels_adjusted = labels.clone()
                        labels_adjusted[~valid_labels_mask] = 0 
                        label_probabilities = torch.gather(softmax_logits, dim=-1, index=labels_adjusted.unsqueeze(-1))

                        # Calculate mean confidence
                        summed_confidence = torch.sum(label_probabilities.squeeze(-1) * valid_labels_mask_float, dim=1)
                        mean_confidence_batch =  summed_confidence / amount_masked
                        confidence[insert_idx : insert_idx_batch] = mean_confidence_batch.detach().cpu().numpy()
        
                        # Calculate entropy
                        log_probs = torch.log(softmax_logits + 1e-7) 
                        entropy_batch = -torch.sum(softmax_logits * log_probs, dim=-1)
                        mean_entropy_batch = torch.sum(entropy_batch * valid_labels_mask_float, dim=1) / amount_masked
                        entropy[insert_idx : insert_idx_batch] = mean_entropy_batch.detach().cpu().numpy()

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
                    
                    if args.get_dynamics:
                        if save_thread is not None:
                            save_thread.join()

                        offset = (step + 1 - args.logging_steps) * args.train_batch_size
                        
                        correctness_copy = correctness.copy()
                        confidence_copy = confidence.copy()
                        entropy_copy = entropy.copy()

                        save_thread = threading.Thread(target=async_save_dynamics, 
                                                    args=(args, args.dynamics_path, offset, correctness_copy, confidence_copy, entropy_copy, args.train_batch_size))
                        save_thread.start()

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
            checkpoint_name = f"checkpoint-{global_step:08d}"
            ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            save_model(args, ckpt_dir, checkpoint_name, model, tokenizer, optimizer, scheduler, scaler)
            if save_thread is not None:
                save_thread.join()
            break
        
        # Always end after single epoc
        if save_thread is not None:
            save_thread.join()

    tb_writer.close()

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
        inputs, labels = mask_tokens(batch, tokenizer, args)
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

def compute_ckpt_dynamics(args, train_dataset, instances, ckpt_nr, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix=""):
    set_seed(args.seed) 
    def collate(examples: List[Tuple[torch.Tensor, torch.Tensor]]):
        tokens, masks = zip(*examples)  
        
        if tokenizer._pad_token is None:
            padded_tokens = pad_sequence(tokens, batch_first=True)
        else:
            padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
        
        padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)
        return padded_tokens, padded_masks
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    train_sampler = CustomSampler(instances)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, shuffle=False, num_workers=2,
        batch_size=args.eval_batch_size, collate_fn=collate, pin_memory=True
    )

    correctness = np.zeros((args.logging_steps * args.eval_batch_size), dtype=np.float32)
    confidence = np.zeros((args.logging_steps * args.eval_batch_size), dtype=np.float32)
    entropy = np.zeros((args.logging_steps * args.eval_batch_size), dtype=np.float32)

    print()
    print(instances)
    print()
    save_thread = None
    for step, batch in enumerate(tqdm(train_dataloader, desc="Evaluating", ncols=100)):
    
        inputs, labels = mask_inputs(batch[0], batch[1], tokenizer)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        # If some of the input is padded, then the attention mask is needed
        attention_mask = (inputs != tokenizer.pad_token_id)  # word_tokens --> 1, pad_token --> 0
        if attention_mask.all():
            attention_mask = None

        with torch.no_grad():
            with autocast():
                outputs = model(inputs, attention_mask=attention_mask, masked_lm_labels=labels)
    
            logits = outputs['logits']
            softmax_logits = torch.softmax(logits, dim=-1)  

            insert_idx = (step % args.logging_steps) * args.eval_batch_size 
            insert_idx_batch = insert_idx + args.eval_batch_size

            valid_labels_mask = labels != -100
            valid_labels_mask_float = valid_labels_mask.float()
            amount_masked = torch.sum(valid_labels_mask_float, dim=1)
            # Avoid division by 0 in case no words are masked
            amount_masked = amount_masked.masked_fill_(amount_masked == 0, 1.)

            # Calculate correctness per instance
            predicted_token_ids = torch.argmax(logits, dim=-1)
            correct_predictions_mask = (predicted_token_ids == labels) & valid_labels_mask
            correct_predictions = torch.sum(correct_predictions_mask.float(), dim=1)
            correctness_batch = correct_predictions / amount_masked
            correctness[insert_idx : insert_idx_batch] = correctness_batch.detach().cpu().numpy()

            # Calculate mean confidence
            labels_adjusted = labels.clone()
            labels_adjusted[~valid_labels_mask] = 0 
            label_probabilities = torch.gather(softmax_logits, dim=-1, index=labels_adjusted.unsqueeze(-1))
            summed_confidence = torch.sum(label_probabilities.squeeze(-1) * valid_labels_mask_float, dim=1)
            mean_confidence_batch =  summed_confidence / amount_masked
            confidence[insert_idx : insert_idx_batch] = mean_confidence_batch.detach().cpu().numpy()

            # Calculate entropy
            log_probs = torch.log(softmax_logits + 1e-7) 
            entropy_batch = -torch.sum(softmax_logits * log_probs, dim=-1)
            mean_entropy_batch = torch.sum(entropy_batch * valid_labels_mask_float, dim=1) / amount_masked
            entropy[insert_idx : insert_idx_batch] = mean_entropy_batch.detach().cpu().numpy()

        if (step + 1) % args.logging_steps == 0 and step > 0:
            
            if save_thread is not None:
                save_thread.join()

            offset = (step + 1 - args.logging_steps) * args.eval_batch_size
            
            correctness_copy = correctness.copy()
            confidence_copy = confidence.copy()
            entropy_copy = entropy.copy()

            save_thread = threading.Thread(target=async_save_dynamics, 
                                           args=(args, args.dynamics_path, offset, correctness_copy, confidence_copy, entropy_copy, args.eval_batch_size, ckpt_nr))
            save_thread.start()
        
        if step + 1 >= args.max_steps:
            break
    
    if save_thread is not None:
        save_thread.join()

def compute_dynamics(args, train_dataset, tokenizer):

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')

    checkpoint_names = []
    model_names = []
    if os.path.isdir(ckpt_dir):
        checkpoint_names = [fn for fn in os.listdir(ckpt_dir) if fn.startswith('checkpoint-')]
        logger.info(f"Checkpoints detected: {len(checkpoint_names)}")
    if len(checkpoint_names) > 0:
        checkpoint_names = sorted(checkpoint_names, key=lambda p: int(p.split('-')[-1]))
        for i, ckpt_name in enumerate(checkpoint_names):
            model_names.append(os.path.join(ckpt_dir, ckpt_name))
    else:
        logger.warning(f"No checkpoint detected: {ckpt_dir}")
        return -1

    assert args.block_size <= tokenizer.model_max_length

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "Why do you want the default config?? Please use --config_name or --model_name_or_path"
        )

    args.instance_data_path = os.path.join(args.output_dir, "instance_data.hdf5")
    with h5py.File(args.instance_data_path, 'r') as f:
        instances = f['instance_order'][:]

    logger.info(f"Computing dynamics from: {args.instance_data_path}")

    instance_amount = len(instances) 

    for i in args.dynamics_ckpts_list:

        with h5py.File(args.dynamics_path, 'a') as f:
            dataset_names = [
                f"confidence_ckpt_{i}",
                f"correctness_ckpt_{i}",
                f"entropy_ckpt_{i}"
            ]

            for dataset_name in dataset_names:
                if dataset_name in f:
                    del f[dataset_name]
                
                f.create_dataset(dataset_name, shape=(instance_amount), dtype=np.float32)

        if i < len(model_names) and i > 0:
            args.model_name_or_path = model_names[i]
        else:
            logger.warning(f"Invalid checkpoint chosen. Valid Checkpoints: 1-{len(model_names)}")
        logger.info(f"Evaluating {args.model_name_or_path} for {args.max_steps} steps.")

        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            args=args
        )
        model.to(args.device)
        compute_ckpt_dynamics(args, train_dataset, instances, i, model, tokenizer)

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
    set_seed(args.seed)
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

    logger.info(model)
    return model, tokenizer

def main():
    parser = process_args()
    args = parser.parse_args()
    args.mask_path = os.path.join(args.mask_path, f"mask-set-{args.mask_set}.hdf5")
    
    if args.dynamics_path is None:
        args.dynamics_path = os.path.join(args.output_dir, f"dynamics_data_{args.mask_set}.hdf5")
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    model, tokenizer = get_model_tokenizer(args)

    if args.generate_masks:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        generate_masks(args, train_dataset, tokenizer)
        args.generate_masks = False

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        model.to(args.device)
        train(args, train_dataset, model, tokenizer)

    if args.compute_dynamics:  
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        compute_dynamics(args, train_dataset, tokenizer)

if __name__ == "__main__":
    main()
