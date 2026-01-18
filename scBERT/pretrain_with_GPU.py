# -*- coding: utf-8 -*-
import argparse
import gc
import math
from functools import reduce

import scanpy as sc
import torch
import tqdm
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

# Import local utilities
from . import utils
from .performer_pytorch import PerformerLM


# --- Dataset Logic ---
class SCDataset(Dataset):
    def __init__(self, data, bin_num, device):
        super().__init__()
        self.data = data
        self.bin_num = bin_num
        self.device = device

    def __getitem__(self, index):
        # Handle sparse matrix if necessary
        full_seq = (
            self.data[index].toarray()[0]
            if hasattr(self.data[index], "toarray")
            else self.data[index]
        )

        # Binning: cap at bin_num, reserve 0 for padding/special
        full_seq[full_seq > self.bin_num] = self.bin_num
        full_seq = torch.from_numpy(full_seq).long()

        # Append special token (e.g. [CLS])
        full_seq = torch.cat((full_seq, torch.tensor([0])))

        return full_seq.to(self.device)

    def __len__(self):
        return self.data.shape[0]


# --- Masking Logic (Self-Supervised Pretraining) ---
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len = mask.shape
    device = mask.device
    max_masked = math.ceil(prob * seq_len)
    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = torch.arange(seq_len, device=device).repeat(batch, 1)
    mask_excess = (mask_excess >= (num_tokens * prob).ceil())[:, :max_masked]
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)
    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


def data_mask(
    data,
    mask_prob,
    replace_prob,
    num_tokens,
    random_token_prob,
    mask_token_id,
    pad_token_id,
    mask_ignore_token_ids,
):
    mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])
    # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
    # also do not include these special tokens in the tokens chosen at random
    no_mask = mask_with_tokens(
        data, mask_ignore_token_ids
    )  # ignore_token as True, will not be masked later
    mask = get_mask_subset_with_prob(
        ~no_mask, mask_prob
    )  # get the True/False mask matrix
    # get mask indices
    ## mask_indices = torch.nonzero(mask, as_tuple=True)   # get the index of mask(nonzero value of mask matrix)
    # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
    masked_input = data.clone().detach()
    # if random token probability > 0 for mlm
    if random_token_prob > 0:
        assert num_tokens is not None, (
            "num_tokens keyword must be supplied when instantiating MLM if using random token replacement"
        )
        random_token_prob = prob_mask_like(
            data, random_token_prob
        )  # get the mask matrix of random token replace
        random_tokens = torch.randint(
            0, num_tokens, data.shape, device=data.device
        )  # generate random token matrix with the same shape as input
        random_no_mask = mask_with_tokens(
            random_tokens, mask_ignore_token_ids
        )  # not masked matrix for the random token matrix
        random_token_prob &= (
            ~random_no_mask
        )  # get the pure mask matrix of random token replace
        random_indices = torch.nonzero(
            random_token_prob, as_tuple=True
        )  # index of random token replace
        masked_input[random_indices] = random_tokens[
            random_indices
        ]  # replace some tokens by random token
    # [mask] input
    replace_prob = prob_mask_like(
        data, replace_prob
    )  # get the mask matrix of token being masked
    masked_input = masked_input.masked_fill(
        mask * replace_prob, mask_token_id
    )  # get the data has been masked by mask_token
    # mask out any tokens to padding tokens that were not originally going to be masked
    labels = data.masked_fill(~mask, pad_token_id)  # the label of masked tokens
    return masked_input, labels


# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bin_num", type=int, default=5, help="Number of bins.")
    parser.add_argument("--gene_num", type=int, default=16906, help="Number of genes.")
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=2021, help="Random seed.")
    parser.add_argument(
        "--batch_size", type=int, default=3, help="Number of batch size."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate."
    )
    parser.add_argument(
        "--grad_acc", type=int, default=60, help="Number of gradient accumulation."
    )
    parser.add_argument(
        "--valid_every",
        type=int,
        default=1,
        help="Number of training epochs between twice validation.",
    )
    parser.add_argument(
        "--mask_prob", type=float, default=0.15, help="Probability of masking."
    )
    parser.add_argument(
        "--replace_prob",
        type=float,
        default=0.9,
        help="Probability of replacing with [MASK] token for masking.",
    )

    parser.add_argument(
        "--pos_embed",
        action="store_true",
        help="Use Gene2vec encoding."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/panglao_human.h5ad",
        help="Path of data for pretraining.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./ckpts/",
        help="Directory of checkpoint to save.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="panglao_pretrain",
        help="Pretrained model name.",
    )

    args = parser.parse_args()

    print(args)

    # Hardware & OS Adaptivity
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    SEED = args.seed
    EPOCHS = args.epoch
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION = args.grad_acc
    LEARNING_RATE = args.learning_rate
    SEQ_LEN = args.gene_num + 1
    VALIDATE_EVERY = args.valid_every
    CLASS = args.bin_num + 2
    MASK_PROB = args.mask_prob
    REPLACE_PROB = args.replace_prob
    RANDOM_TOKEN_PROB = 0.0
    MASK_TOKEN_ID = CLASS - 1
    PAD_TOKEN_ID = CLASS - 1
    MASK_IGNORE_TOKEN_IDS = [0]
    POS_EMBED_USING = args.pos_embed

    model_name = args.model_name
    ckpt_dir = args.ckpt_dir

    utils.seed_all(SEED)

    # Data Prep
    adata = sc.read_h5ad(args.data_path)
    data_train, data_val = train_test_split(
        adata.X, test_size=0.05, random_state=args.seed
    )

    del adata
    gc.collect()

    train_ds = SCDataset(data_train, args.bin_num, device)
    val_ds = SCDataset(data_val, args.bin_num, device)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        # sampler=DistributedSampler(train_ds)
    )

   # val_sampler = SequentialDistributedSampler(val_ds)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
       # sampler=val_sampler,
    )

    # Model Init
    model = PerformerLM(
        num_tokens=CLASS,
        dim=200,
        depth=6,
        max_seq_len=SEQ_LEN,
        heads=10,
        local_attn_heads=0,
        g2v_position_emb=POS_EMBED_USING,
    ).to(device)

    # optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # learning rate scheduler
    scheduler = utils.CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=LEARNING_RATE,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9,
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction="mean").to(
        device
    )

    softmax = nn.Softmax(dim=-1)

    for i in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        cum_acc = 0.0

        scaler = torch.amp.GradScaler(device.type)
        optimizer.zero_grad()

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {i}")
        for index, data in enumerate(pbar):
            index += 1
            data = data.to(device)
            data, labels = data_mask(
                data,
                MASK_PROB,
                REPLACE_PROB,
                None,
                RANDOM_TOKEN_PROB,
                MASK_TOKEN_ID,
                PAD_TOKEN_ID,
                MASK_IGNORE_TOKEN_IDS,
            )
            with torch.autocast(device.type, dtype=torch.float16):
                logits = model(data)
                loss = loss_fn(
                    logits.transpose(1, 2), labels
                ) / GRADIENT_ACCUMULATION

            scaler.scale(loss).backward()

            if index % GRADIENT_ACCUMULATION == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running_loss += loss.item()
            final = softmax(logits)[..., 1:-1]
            final = final.argmax(dim=-1) + 1
            pred_num = (labels != PAD_TOKEN_ID).sum(dim=-1)
            correct_num = ((labels != PAD_TOKEN_ID) * (final == labels)).sum(dim=-1)
            cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
        epoch_loss = running_loss / index
        epoch_acc = 100 * cum_acc / index
        print(
            f"== Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  =="
        )
        scheduler.step()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            running_loss = 0.0
            predictions = []
            truths = []
            with torch.autocast(device.type, dtype=torch.float16):
                for index, data in enumerate(val_loader):
                    index += 1
                    data = data.to(device)
                    data, labels = data_mask(
                        data,
                        MASK_PROB,
                        REPLACE_PROB,
                        None,
                        RANDOM_TOKEN_PROB,
                        MASK_TOKEN_ID,
                        PAD_TOKEN_ID,
                        MASK_IGNORE_TOKEN_IDS,
                    )
                    logits = model(data)
                    loss = loss_fn(logits.transpose(1, 2), labels)
                    running_loss += loss.item()
                    softmax = nn.Softmax(dim=-1)
                    final = softmax(logits)[..., 1:-1]
                    final = final.argmax(dim=-1) + 1
                    predictions.append(final)
                    truths.append(labels)
                del data, labels, logits, final

                predictions = torch.cat(predictions, dim=0)
                truths = torch.cat(truths, dim=0)

                correct_num = (
                    ((truths != PAD_TOKEN_ID) * (predictions == truths))
                    .sum(dim=-1)[0]
                    .item()
                )
                val_num = (truths != PAD_TOKEN_ID).sum(dim=-1)[0].item()
                val_loss = running_loss / index
 
            val_acc = 100 * correct_num / val_num
            print(
                f"== Epoch: {i} | Validation Loss: {val_loss:.6f} | Accuracy: {val_acc:6.4f}%  =="
            )
        del predictions, truths

        if i % 2 == 0:
            utils.save_ckpt(i, model, optimizer, scheduler, epoch_loss, model_name, ckpt_dir)

if __name__ == "__main__":
    main()
