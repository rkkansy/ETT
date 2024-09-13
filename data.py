import copy
import os
import random
import numpy as np

import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Sampler

class CustomSampler(Sampler):
    def __init__(self, indices):

        self.indices = indices
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)

def async_save_dynamics(args, filename, offset, correctness, confidence, entropy, batch_size, eval_epoch=0):

    with h5py.File(filename, 'a') as f: 
        for i in range(len(correctness)):
            f[f"correctness_ckpt_{eval_epoch}"][i + offset] = correctness[i]
            f[f"confidence_ckpt_{eval_epoch}"][i + offset] = confidence[i]
            f[f"entropy_ckpt_{eval_epoch}"][i + offset] = entropy[i]
    
    print()
    print(f"Saved Batches for checkpoint: {eval_epoch} | {offset // batch_size} - {args.logging_steps * args.gradient_accumulation_steps + offset // batch_size}")

def async_save_masks(filename, offset, masks):
    with h5py.File(filename, 'a') as f:
        if offset + len(masks) > f["masks"].shape[0]:
            print("Error: Trying to write beyond the dataset limits.")
            return

        for i in range(len(masks)):
            f[f"masks"][offset + i , :] = masks[i]


class CoLDataset(Dataset):
    IGNORE_ID = -100
    sent_strategy = 'first'

    def __init__(self, file_path, tokenizer_name, tokenizer, mask_path=None, block_size=512,
                 split_sent=False, voken_dir=None, suffix=None, verbose=False,
                 voken_ablation=None):

        # Open token's hdf5
        token_path = file_path + '.' + tokenizer_name + '.hdf5'
        print(token_path)
        assert os.path.isfile(token_path)
        if verbose:
            print("-------- Load Data -------")
            print("Load tokens from", token_path)
        self.token_hdf5 = h5py.File(token_path, 'r')
        self.tokenizer = tokenizer
        self.tokens = self.token_hdf5['tokens']
        self.masks = None if mask_path is None else h5py.File(mask_path, 'r')['masks']
        self.verbose = verbose
        self._iter_cnt = 0

        # Split for every block_size tokens
        # The last block without full length will be dropped.
        num_tokens = len(self.tokens)
        self.starts = list(range(0, num_tokens, block_size))
        self.batches = list(zip(self.starts[:-1], self.starts[1:]))

        manual_filtered =False
        if "en.train.raw" in file_path and tokenizer_name == "bert-base-uncased":
            self.batches = manual_filter(self.batches)
            if verbose:
                print("Data: Mannually filter the range for counties.")
            manual_filtered = True

        # batch_info
        if verbose:
            print("Split sent with block size", block_size)
            print(f"Total batches: {len(self.batches)}")
            print(f"Total tokens: {len(self.tokens)}")
            if voken_dir is not None:
                print(f"Total vokens: {len(self.vokens)}")
            if voken_ablation is not None:
                print("The model will process voken ablation strategy:", voken_ablation)
            print()

        block_check(self.batches, block_size, fixed_size=True, manual_filtered=manual_filtered)

    def __len__(self):
        return len(self.batches) - 1

    def __getitem__(self, item):
        token_start, token_end = self.batches[item]
        if self._iter_cnt < 5 and self.verbose:
            #print(f"Data Loader: data iteration {self._iter_cnt}, with range {token_start} to {token_end}.")
            self._iter_cnt += 1
        tokens = list(self.tokens[token_start: token_end])
        token_tensor = torch.tensor(
            self.tokenizer.build_inputs_with_special_tokens(tokens),
            dtype=torch.long)

        if self.masks is None:
            return token_tensor
        else:
            return token_tensor, torch.tensor(self.masks[item], dtype=torch.long)

    def get_item_info(self, item):
        token_start = self.batches[item]
        token_end = self.batches[item + 1]
        return token_start, token_end

    def __del__(self):
        self.token_hdf5.close()


FORBIDDEN_RANGE = (
    119314944,      # Start of iter 3700
    187053048       # End of iter 5800
)


def intersect(x, y):
    x1, x2 = x
    y1, y2 = y
    if x2 <= y1 or x2 >= y2:
        # Case 1: [   x    )[   y    )
        # Case 2: [   y    )[   x    )
        return False
    return True


def manual_filter(batches):
    batches = list(filter(
        lambda x: not intersect(x, FORBIDDEN_RANGE),
        batches
    ))
    return batches


def block_check(batches, block_size, fixed_size=False, manual_filtered=False):
    """
    Check whether the batches satisfy following requirements.
        1. Monotonic
        2. Mutually exclusive
        3. Range < block_size
    """
    last_end = 0
    for start_token, end_token in batches:
        assert last_end <= start_token
        if fixed_size:
            assert (end_token - start_token) == block_size, 'len([%d, %d)) != %d' % (start_token, end_token, block_size)
        else:
            assert (end_token - start_token) <= block_size, 'len([%d, %d)) > %d' % (start_token, end_token, block_size)
        if manual_filtered:
            assert not intersect((start_token, end_token), FORBIDDEN_RANGE)
        last_end = end_token



