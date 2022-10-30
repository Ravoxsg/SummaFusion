import torch
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import T5TokenizerFast, BartTokenizerFast, PegasusTokenizerFast
from models.model_bart_source_5b import BartForConditionalGenerationSource5b



def build_tokenizer(args):
    tokenizer = None
    if args.model_type.startswith("t5"):
        print("\nUsing T5 tokenizer")
        tokenizer = T5TokenizerFast.from_pretrained(args.model, cache_dir = args.cache_dir)
    elif args.model_type.startswith("bart"):
        print("\nUsing BART tokenizer")
        tokenizer = BartTokenizerFast.from_pretrained(args.model, cache_dir = args.cache_dir)
    elif args.model_type.startswith("pegasus"):
        print("\nUsing Pegasus tokenizer")
        tokenizer = PegasusTokenizerFast.from_pretrained(args.model, cache_dir = args.cache_dir)

    return tokenizer


def build_model(args):
    model = BartForConditionalGenerationSource5b.from_pretrained(args.model, args = args, cache_dir = args.cache_dir)

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
