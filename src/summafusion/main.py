# Train a supervised-reranker.

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import datasets
import time
import pickle
import sys

sys.path.append("/data/mathieu/2nd_stage_summarization/")

from torch.utils.data.dataloader import DataLoader
from transformers import Adafactor, get_linear_schedule_with_warmup

from common.utils import seed_everything, check_scores, check_training_data
from common.data_scored import load_data
from utils import *
from dataset import AbstractiveFusionDataset
from training_utils import *
from model import ModelAbstractiveFusion
from engine import training_loop



parser = argparse.ArgumentParser()

root = "/data/mathieu/"
#root = "/home/ravox/Documents/"

parser.add_argument('--seed', type=int, default = 42)
parser.add_argument('--cuda', type=bool, default = True)
parser.add_argument('--fp16', type=bool, default = True)
parser.add_argument('--debug', type = bool, default = False)
parser.add_argument('--few_shot', type = bool, default = True)
few_shot_size = 100
few_shot_seed = 42
parser.add_argument('--few_shot_size', type=int, default = few_shot_size)
parser.add_argument('--few_shot_seed', type=int, default = few_shot_seed)

# data
parser.add_argument('--dataset_name', type=str, default = "samsum")
parser.add_argument('--generation_method_str', type = str, default = "diverse_beam_search")
parser.add_argument('--scoring_methods_str', type=str, default = "rouge_1+rouge_2+rouge_l")
parser.add_argument('--sep_symbol', type=str, default = "[SEP]")
# train
parser.add_argument('--max_train_size', type=int, default = 1000000) # [287113, 204045, 33704]
# val
parser.add_argument('--val_dataset', type=str, default="few_shot_default_val",
                    choices=["val", "few_shot_default_val"])
parser.add_argument('--max_val_size', type=int, default = 1000000) # in [13368, 11332, 4213]
parser.add_argument("--check_training_data", type = bool, default = True)
# base model
parser.add_argument('--base_model_name', type=str, default = "pegasus")
parser.add_argument('--num_beams', type=int, default = 15)

# training
# input
parser.add_argument('--max_source_length', type=int, default = 1024)
parser.add_argument('--n_candidates', type=int, default = 15)
parser.add_argument('--use_source', type=bool, default = True) 
parser.add_argument('--use_candidates', type=bool, default = True) 
# ordering
parser.add_argument('--encode_position', type = bool, default = True)
parser.add_argument('--full_position_encoding', type = bool, default = False)
parser.add_argument('--position_symbol', type=str, default = "CAND_")
parser.add_argument('--encode_generation_method', type=bool, default = False)
# subsetting
parser.add_argument('--n_candidates_to_use', type=int, default = 15)
# subsampling
parser.add_argument('--source_dropout', type=bool, default = True)
parser.add_argument('--source_dropout_prob', type=float, default = 0.2)
parser.add_argument('--source_dropout_at_inference', type=bool, default = False)
parser.add_argument('--n_subsample_low', type=int, default = 2)
parser.add_argument('--n_subsample_high', type=int, default = 15)
parser.add_argument('--subsample_at_inference', type=bool, default = False)
# shuffling
parser.add_argument('--shuffle_candidates', type=bool, default = False)

# fusion model
# general
parser.add_argument('--model', type=str, default = "facebook/bart-large")
parser.add_argument('--cache_dir', type=str, default = root + "hf_models/bart-large/")
parser.add_argument('--hidden_size', type=int, default = 768) # 768 / 1024
# classification
parser.add_argument('--classify_candidates', type = bool, default = True)
parser.add_argument('--cls_loss_weight', type = float, default = 1.0)
parser.add_argument('--cls_hidden_size', type = int, default = 2048)
parser.add_argument('--subsample_cls_cands', type = bool, default = True)
parser.add_argument('--n_subsample_cls_neg', type = int, default = 1)
parser.add_argument('--use_source_for_cls', type = bool, default = True)
parser.add_argument('--use_ss_for_cls', type = bool, default = False)

# optimization
parser.add_argument('--shuffle_train', type=bool, default = True)
parser.add_argument('--n_epochs', type=int, default = 5)
parser.add_argument('--adafactor', type=bool, default = False)
parser.add_argument('--train_bs', type=int, default = 2)
parser.add_argument('--inference_bs', type=int, default = 2) # 2 for large, 4 for base
parser.add_argument('--gradient_accumulation_steps', type=int, default = 32) # use effective_batch_size = 64
parser.add_argument('--lr', type=float, default = 2e-5)
parser.add_argument('--wd', type=float, default = 0)
parser.add_argument('--gradient_clipping', type=float, default = 10e10)
parser.add_argument('--scheduler', type=str, default = "constant")  # in ["constant", "linear"]
parser.add_argument('--warmup_ratio', type=float, default = 0.05)
parser.add_argument('--early_stopping_metric', type=str, default = "mean_r") # in ["mean_r", "r1", "r2", "rl"]
parser.add_argument('--print_every', type = int, default = 100)
parser.add_argument('--eval_per_epoch', type = bool, default = False)

# summary generation
parser.add_argument('--num_gen_beams', type = int, default = 10) # default: 15
parser.add_argument('--num_gen_beam_groups', type = int, default = 1) # default: 15
parser.add_argument('--repetition_penalty', type = float, default = 1.0)
parser.add_argument('--length_penalty', type = float, default = 1.0)

# evaluation
parser.add_argument('--eval_epoch_0', type = bool, default = True)
parser.add_argument('--stemmer', type = bool, default = True)

# export
parser.add_argument('--save_model', type = bool, default = True)
parser.add_argument('--save_model_path', type = str, default = "saved_models/samsum_few_shot/few_shot_100_seed_42/pytorch_model.bin".format(few_shot_size, few_shot_seed))
parser.add_argument('--load_model', type = bool, default = False)
parser.add_argument('--load_model_path', type = str, default = "")

args = parser.parse_args()
args.generation_methods = args.generation_methods_str.split("+")
args.scoring_methods = args.scoring_methods_str.split("+")
args.n_tasks = len(args.scoring_methods)

dataset_names = ["xsum", "reddit", "samsum"]
folder_names = ["XSum", "Reddit", "SAMSum"]
train_sizes = [[102000, 102045], [17000, 16704], [7350, 7382]]
val_sizes = [13368, 11332, 4213, 818]
train_model_names = [
    ["pegasus_xsum_second_half_shuffled_1", "pegasus_xsum_first_half_shuffled_1"],
    ["pegasus_reddit_second_half_shuffled_1", "pegasus_reddit_first_half_shuffled_1"],
    ["pegasus_samsum_second_half_shuffled_2", "pegasus_samsum_first_half_shuffled_2"],
]
model_names = ["pegasus_xsum", "pegasus_reddit_train_1", "pegasus_samsum_train_4"]
max_canditate_lengths_95 = [34, 43, 42]
max_summary_lengths = [64, 64, 64]
max_gen_summary_lengths = [64, 64, 64]
no_repeat_ngram_sizes = [3, 3, 3]
eval_every = [500, 100, 50]

idx = dataset_names.index(args.dataset_name)
args.data_folder = root + "DATASETS/{}/data/".format(folder_names[idx])
args.scored_summaries_path = "../reranking_data/{}/".format(folder_names[idx])
args.train_datasets = ["first_half_train_shuffled", "second_half_train_shuffled"]
args.train_sizes = train_sizes[idx]
args.val_size = val_sizes[idx]
args.train_model_names = train_model_names[idx]
args.model_name = model_names[idx]
args.max_candidate_length = max_canditate_lengths_95[idx]
args.max_summary_length = max_summary_lengths[idx] # ground truth
args.max_gen_summary_length = max_gen_summary_lengths[idx] # fusion summary
args.no_repeat_ngram_size = no_repeat_ngram_sizes[idx]
args.eval_every = eval_every[idx]

train_model_names_few_shot = [
    ["pegasus_xsum_second_half_train_{}_seed_{}_shuffled_1".format(args.few_shot_size, args.few_shot_seed), "pegasus_xsum_first_half_train_{}_seed_{}_shuffled_1".format(args.few_shot_size, args.few_shot_seed)],
    ["pegasus_reddit_second_half_train_{}_seed_{}_shuffled_1".format(args.few_shot_size, args.few_shot_seed), "pegasus_reddit_first_half_train_{}_seed_{}_shuffled_1".format(args.few_shot_size, args.few_shot_seed)],
    ["pegasus_samsum_second_half_train_{}_seed_{}_shuffled_1".format(args.few_shot_size, args.few_shot_seed), "pegasus_samsum_first_half_train_{}_seed_{}_shuffled_1".format(args.few_shot_size, args.few_shot_seed)]
]
model_names_few_shot = [
    "pegasus_xsum_train_{}_seed_{}_1".format(args.few_shot_size, args.few_shot_seed), 
    "pegasus_reddit_train_{}_seed_{}_1".format(args.few_shot_size, args.few_shot_seed), 
    "pegasus_samsum_train_{}_seed_{}_1".format(args.few_shot_size, args.few_shot_seed)
]
if args.few_shot:
    args.train_datasets = ["first_half_train_train_{}_seed_{}_shuffled".format(args.few_shot_size, args.few_shot_seed),
                           "second_half_train_{}_seed_{}_shuffled".format(args.few_shot_size, args.few_shot_seed)]
    if args.val_dataset == "few_shot_default_val":
        args.val_dataset = "val_{}_seed_{}".format(args.few_shot_size, args.few_shot_seed)
    args.n_epochs = 30
    args.eval_every = 1000000
    args.eval_per_epoch = True
    if args.few_shot_size == 10:
        args.train_sizes = [5, 5]
        args.gradient_accumulation_steps = 2
    elif args.few_shot_size == 100:
        args.train_sizes = [50, 50]
        args.gradient_accumulation_steps = 8
    elif args.few_shot_size == 1000:
        args.train_sizes = [500, 500]
        args.gradient_accumulation_steps = 32
    args.val_size = min(val_sizes[idx], args.few_shot_size)
    args.train_model_names = train_model_names_few_shot[idx]
    args.model_name = model_names_few_shot[idx]

print("*" * 50)
print(args)



def main(args):
    # seed
    seed_everything(args.seed)

    # device
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    args.device = device
    print("Using device: {}".format(device))

    # tokenizer
    tokenizer = build_tokenizer(args)
    if args.encode_position and args.position_symbol != "":
        if args.full_position_encoding:
            for j in range(int(len(args.generation_methods) * args.num_beams)):
                tokenizer.add_tokens(["{}{}".format(args.position_symbol, j)])
        else:
            for j in range(args.num_beams):
                tokenizer.add_tokens(["{}{}".format(args.position_symbol, j)])
    if args.encode_generation_method:
        for j in range(len(args.generation_methods)):
            tokenizer.add_tokens(["GEN_{}".format(j)])
    
    # check data
    if args.check_training_data:
        check_training_data(args)

    # data & datasets
    datasets = []
    for x in [(args.train_datasets, args.train_sizes), (args.val_dataset, args.val_size)]:
        set, size = x
        # data
        train = set == args.train_datasets
        texts, summaries, scored_summaries = load_data(set, size, args, individual_txt = args.highlights, train = train)
        print(set, len(texts))
        if args.debug:
            texts = texts[:10]
            summaries = summaries[:10]
            scored_summaries = scored_summaries[:10]
            print(set, len(texts), len(summaries))
        print("loaded new data!", len(texts), len(summaries), len(scored_summaries), len(scored_summaries[0]),
              len(scored_summaries[0][0]), len(scored_summaries[0][1]), len(scored_summaries[0][1][0]))
        # dataset
        mode = "train"
        if not (train):
            mode = "val"
        dataset = AbstractiveFusionDataset(mode, tokenizer, texts, summaries, scored_summaries, args)
        datasets.append(dataset)
        print("There are {} {} batches".format(int(len(dataset.texts) / args.train_bs), set))
    train_dataset = datasets[0]
    train_dataset.texts = train_dataset.texts[:args.max_train_size]
    train_dataset.scored_summaries = train_dataset.scored_summaries[:args.max_train_size]
    train_dataset.labels = train_dataset.labels[:args.max_train_size]
    val_dataset = datasets[1]
    val_dataset.texts = val_dataset.texts[:args.max_val_size]
    val_dataset.scored_summaries = val_dataset.scored_summaries[:args.max_val_size]
    val_dataset.labels = val_dataset.labels[:args.max_val_size]

    print(train_dataset.texts[0])
    print("*"*30)
    print(val_dataset.texts[0])

    # check oracle
    m_train_score = check_scores(train_dataset)
    m_val_score = check_scores(val_dataset)
    print("Oracle - train: {:.4f}, val: {:.4f}".format(m_train_score, m_val_score))

    # data loaders
    train_loader = DataLoader(train_dataset, batch_size = args.train_bs, shuffle = args.shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size = args.inference_bs, shuffle = False)

    # model
    pretrained_model = build_model(args)
    if args.init_source_attn_weights:
        state_dict = {}
        for k in pretrained_model.state_dict().keys():
            if "source_attn" in k:
                start = 0
                for i in range(len(k)):
                    if k[i:(i + 11)] == "source_attn":
                        start = i
                        break
                pre = k[:i]
                post = k[(i + 11):]
                new_k = pre + "self_attn" + post
                state_dict[k] = pretrained_model.state_dict()[new_k]
            else:
                state_dict[k] = pretrained_model.state_dict()[k]
        pretrained_model.load_state_dict(state_dict)
        print("Initialized the source attention weights with the self attention ones")
        #print(pretrained_model.state_dict()["model.decoder.layers.5.source_attn.out_proj.bias"])
        #print(pretrained_model.state_dict()["model.decoder.layers.5.self_attn.out_proj.bias"])
    model = ModelAbstractiveFusion(pretrained_model, tokenizer, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nThe model has {} trainable parameters".format(n_params))
    model = model.to(device)

    if args.encode_position:
        print("Model embeddings size before", model.model.config.vocab_size, model.model.model.shared.num_embeddings)
        model.model.resize_token_embeddings(len(tokenizer))
        print("Model embeddings size after", model.model.config.vocab_size, model.model.model.shared.num_embeddings)
        
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)
    if args.adafactor:
        print("\nUsing Adafactor optimizer")
        optimizer = Adafactor(model.parameters(), lr = args.lr, relative_step = False, weight_decay = args.wd)

    # scheduler
    scheduler = None 
    if args.scheduler == "linear":
        print("\nUsing linear scheduler")
        total_steps = int(args.n_epochs * (len(train_dataset.texts) / (args.gradient_accumulation_steps * args.train_bs)))
        warmup_steps = int(args.warmup_ratio * total_steps)
        print("Total # training steps: {}, # warmup steps: {}".format(total_steps, warmup_steps))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = warmup_steps, 
            num_training_steps = total_steps
        )

    # training loop
    training_loop(train_loader, val_loader, tokenizer, model, optimizer, scheduler, args)



if __name__ == '__main__':
    main(args)
