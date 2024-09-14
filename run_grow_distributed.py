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
from torch.cuda.amp import autocast, GradScaler
from data import CustomSampler
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import h5py

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
)

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from param import process_args
from model import SimpleBertForMaskedLM, SimpleRobertaForMaskedLM

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from ligo import create_ligo_from_model

from run_lm_distributed import load_and_cache_examples, set_seed, mask_tokens, mask_inputs

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "bert": (BertConfig, SimpleBertForMaskedLM, BertTokenizer),
}

def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    set_seed(args.seed)  # Added here for reproducibility

    """ Train the model """

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    tb_writer = SummaryWriter(args.output_dir + '/runs/' + current_time)

    args.train_batch_size = args.train_batch_size

    def collate(examples: List[Tuple[torch.Tensor, torch.Tensor]]):
        tokens, masks = zip(*examples)  
        
        if tokenizer._pad_token is None:
            padded_tokens = pad_sequence(tokens, batch_first=True)
        else:
            padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
        
        padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)
        return padded_tokens, padded_masks

    epoch_size = args.train_batch_size * args.max_steps * args.gradient_accumulation_steps

    if args.instance_data_path:
        if args.data_partition == 'instance_data':
            load_path = os.path.join(args.instance_data_path, f"{args.data_partition}.hdf5")
        else:
            load_path = os.path.join(args.instance_data_path, f"partitions_m-{args.mask_set}_{args.partition_frac}")
            load_path = os.path.join(load_path, f"{args.data_partition}.hdf5")
        with h5py.File(load_path, 'r') as f:
            instances = f['instance_order'][:]
        logger.info(f"Loading instance indices from: {load_path}")
    else:
        instances = list(range(0, len(train_dataset)))
        random.shuffle(instances)

    instances = instances[:epoch_size][::-1]

    train_sampler = CustomSampler(instances)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, shuffle=False, num_workers=0,
        batch_size=args.train_batch_size, collate_fn=collate, pin_memory=True
    )


    t_total = args.max_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = [".bias", "LayerNorm.weight"]
    residual_weights = [".residual_weight", ".residual_bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay + residual_weights)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in residual_weights)],
            "lr": args.learning_rate_res if args.learning_rate_res is not None else args.learning_rate,
            "weight_decay": args.weight_decay_res if args.weight_decay_res is not None else args.weight_decay,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      # betas=(0.9, 0.98),
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    if args.warmup_ratio > 0.:
        assert args.warmup_steps == 0
        args.warmup_steps = int(t_total * args.warmup_ratio)

    print("Optimized with lr %f, steps %d, warmup steps %d, and use beta, epsilon %0.8f." % (
        args.learning_rate, t_total, args.warmup_steps, optimizer.defaults['eps']
    ), optimizer.defaults['betas'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
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
    logger.info("  Num examples = %d", len(train_dataset))
    # logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size  = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_name = os.path.basename(args.model_name_or_path)
            global_step = int(checkpoint_name.split("-")[-1])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from iter %d, epoch %d" % (global_step, epochs_trained))
        except ValueError:
            logger.info("  Do not load model from %s, restart training" % args.model_name_or_path)

    model.zero_grad()

    # IMPORTANT: save the initialization
    if global_step == 0:
        checkpoint_name = f"checkpoint-{global_step:08d}"
        ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        save_model(args, ckpt_dir, checkpoint_name, model, tokenizer, optimizer, scheduler, scaler)

    while True:
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch: {epochs_trained:03d}")
        tr_loss, tr_lm_loss = 0.0, 0.0
        t_start = time.time()
        model.zero_grad()       # Support of accumulating gradients
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
                    current_step=global_step) if args.mlm else model(inputs, labels=labels, current_step=global_step)
            loss = outputs['loss']  # model outputs are always tuple in transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            tr_loss += loss.item()
            tr_lm_loss += outputs['lm_loss'].item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0.:
                    scaler.unscale_(optimizer) 
                    #for group in optimizer.param_groups:
                    #    torch.nn.utils.clip_grad_norm_(group['params'], args.max_grad_norm)
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
                        print(f"train_step={global_step}, train_time={t_elapse}, lr={scheduler.get_last_lr()[0]}, train_loss={train_loss},"
                            f"train_ppl={train_ppl}, eval_ppl={eval_ppl}", file=f)

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

        epochs_trained += 1

    tb_writer.close()

def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    args.eval_batch_size = args.eval_batch_size

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

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
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


def main():
    parser = process_args()
    args = parser.parse_args()

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

             
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args.seed)
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

    args.fuse_init_scheme_depth = args.fuse_init_scheme_width = args.fuse_init_scheme[0]
    if len(args.fuse_init_scheme) >= 2:
        args.fuse_init_scheme_width = args.fuse_init_scheme[1]
    args.fuse_init_noise_depth = args.fuse_init_noise_width = args.fuse_init_noise[0]
    if len(args.fuse_init_noise) >= 2:
        args.fuse_init_noise_width = args.fuse_init_noise[1]

    model = model_class(config=config, args=args)
    model = create_ligo_from_model(model, args)

    if args.model_name_or_path:
        state_dict = torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    args.mask_path = os.path.join(args.mask_path, f"mask-set-{args.mask_set}.hdf5")

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    if args.do_eval:
        result = evaluate(args, model, tokenizer)


if __name__ == "__main__":
    main()
