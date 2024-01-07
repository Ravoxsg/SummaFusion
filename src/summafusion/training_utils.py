import torch
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import T5TokenizerFast, BartTokenizerFast, PegasusTokenizerFast
from modeling_bart import BartForConditionalGeneration


def build_tokenizer(args):
    print("\nUsing BART tokenizer")
    tokenizer = BartTokenizerFast.from_pretrained(args.model, cache_dir = args.cache_dir)

    return tokenizer


def build_model(args):
    model = BartForConditionalGeneration.from_pretrained(args.model, args = args, cache_dir = args.cache_dir)

    return model


def build_optimizer(model, args):
    optimizer = None
    if args.optimizer == "adam":
        print("\nUsing Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "adamw":
        print("\nUsing AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    return optimizer


def build_scheduler(optimizer, train_steps, args):
    scheduler = None
    if args.scheduler == "linear_warmup":
        print("\nUsing linear warmup scheduler")
        warmup_steps = int(args.warmup_ratio * train_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, train_steps)

    return scheduler
