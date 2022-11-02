# Evaluate the performance of a trained re-ranker.

import argparse
import sys
import time
import torch.nn as nn

sys.path.append("/data/mathieu/SummaFusion/src/")

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

from common.utils import seed_everything
from common.evaluation import *
from common.data_scored import load_data
from utils import *
from dataset import AbstractiveFusionDataset
from training_utils import *
from model import ModelAbstractiveFusion
from engine import validate
from evaluation_utils import *



parser = argparse.ArgumentParser()

root = "/data/mathieu/"

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--fp16', type=bool, default=True)
parser.add_argument('--few_shot', type=bool, default=True)
parser.add_argument('--few_shot_size', type=int, default=100)
parser.add_argument('--few_shot_seed', type=int, default=42)

# data
parser.add_argument('--dataset', type=str, default="samsum")
parser.add_argument('--generation_methods_str', type = str, default = "diverse_beam_search")
parser.add_argument('--scoring_methods_str', type=str, default = "rouge_1+rouge_2+rouge_l")
parser.add_argument('--sep_symbol', type=str, default="[SEP]")
# val
parser.add_argument('--val_dataset', type=str, default="few_shot_default_val",
                    choices=["val", "test", "few_shot_default_val"])
parser.add_argument('--max_val_size', type=int, default=50000)
# base model
parser.add_argument('--base_model_name', type=str, default="pegasus")
parser.add_argument('--num_beams', type=int, default=15)

# training
# input
parser.add_argument('--max_source_length', type=int, default=1024)
parser.add_argument('--n_candidates', type=int, default=15)
parser.add_argument('--use_source', type=bool, default=True)
parser.add_argument('--use_candidates', type=bool, default=True)
# ordering
parser.add_argument('--encode_position', type=bool, default=True)
parser.add_argument('--full_position_encoding', type=bool, default=False)
parser.add_argument('--position_symbol', type=str, default="CAND_")
parser.add_argument('--encode_generation_method', type=bool, default=False)
# subsetting
parser.add_argument('--n_candidates_to_use', type=int, default=15)
# subsampling
parser.add_argument('--source_dropout', type=bool, default=False)
parser.add_argument('--source_dropout_prob', type=float, default=0.2)
parser.add_argument('--source_dropout_at_inference', type=bool, default=False)
parser.add_argument('--n_subsample_low', type=int, default=2)
parser.add_argument('--n_subsample_high', type=int, default=15)
parser.add_argument('--subsample_at_inference', type=bool, default=False)
# shuffling
parser.add_argument('--shuffle_candidates', type=bool, default=False)

# fusion model
# general
parser.add_argument('--model', type=str, default="facebook/bart-large")
parser.add_argument('--cache_dir', type=str, default=root + "hf_models/bart-large/")
parser.add_argument('--hidden_size', type=int, default=1024)  # 768 / 1024
# classification
parser.add_argument('--classify_candidates', type=bool, default=True)
parser.add_argument('--cls_loss_weight', type=float, default=1)
parser.add_argument('--cls_hidden_size', type=int, default=2048)  # 1024 / 2048
parser.add_argument('--use_source_for_cls', type=bool, default=True)
parser.add_argument('--use_ss_for_cls', type=bool, default=False)
# weights
parser.add_argument('--load_model', type=bool, default=True)
parser.add_argument('--load_model_path', type=str, default= "few_shot_100_seed_42")
# optimization
parser.add_argument('--inference_bs', type=int, default=4)

# metrics
# 1 - ROUGE
parser.add_argument('--eval_rouge', type=bool, default=True)
# 2 - BERTScore
parser.add_argument('--eval_bertscore', type=bool, default=False)
# 3 - BARTScore
parser.add_argument('--eval_bartscore', type=bool, default=False)
# 4 - Copying
parser.add_argument('--eval_ngram_copying', type=bool, default=False)
# 5 - Abstractiveness
parser.add_argument('--eval_new_ngram', type=bool, default=True)

# summary generation
parser.add_argument('--num_return_sequences', type=int, default=1)  # default: 1
parser.add_argument('--num_gen_beams', type=int, default=10)  # default: 15
parser.add_argument('--repetition_penalty', type=float, default=1.0)  # 1.0
parser.add_argument('--length_penalty', type=float, default=1.0)

# evaluation
parser.add_argument('--stemmer', type=bool, default=True)
parser.add_argument('--n_show_summaries', type=int, default=0)

# evaluation aspects
parser.add_argument('--evaluate_candidates_abstractiveness', type=bool, default=True)
parser.add_argument('--evaluate_new_summaries', type=bool, default=True)
parser.add_argument('--show_distribution_over_candidates', type=bool, default=False)
parser.add_argument('--n_bins', type=int, default=10)
parser.add_argument('--evaluate_per_summary_quality', type=bool, default=False)
parser.add_argument('--evaluate_per_summary_diversity', type=bool, default=False)
parser.add_argument('--evaluate_per_source_length', type=bool, default=False)
parser.add_argument('--evaluate_per_target_ratio', type=bool, default=False)
parser.add_argument('--evaluate_break_oracle', type=bool, default=False)
parser.add_argument('--evaluate_ablation_candidates', type=bool, default=False)
parser.add_argument('--n_ablation_candidates', type=list, default=[0, 1, 3, 5, 10, 15])
parser.add_argument('--evaluate_without_source', type=bool, default=False)

args = parser.parse_args()
args.generation_methods = args.generation_methods_str.split("+")
args.scoring_methods = args.scoring_methods_str.split("+")
args.n_tasks = len(args.scoring_methods)

dataset_names = ["xsum", "reddit", "samsum"]
folder_names = ["XSum", "Reddit", "SAMSum"]
val_sizes = [11332, 4213, 818]
test_sizes = [11334, 4222, 819]
model_names = ["pegasus_xsum", "pegasus_reddit_train_1", "pegasus_samsum_train_4"]
max_canditate_lengths_95 = [34, 43, 42]
max_summary_lengths = [64, 64, 64]
max_gen_summary_lengths = [64, 64, 64]
no_repeat_ngram_sizes = [3, 3, 3]

idx = dataset_names.index(args.dataset)

args.data_folder = root + "../../data/".format(folder_names[idx])
args.scored_summaries_path = "../../scored_summaries/{}/".format(folder_names[idx])
args.generation_methods = args.generation_methods.split(",")
if args.val_dataset == "val":
    args.val_size = val_sizes[idx]
elif args.val_dataset == "test":
    args.val_size = test_sizes[idx]
args.model_name = model_names[idx]
args.max_candidate_length = max_canditate_lengths_95[idx]
args.max_summary_length = max_summary_lengths[idx]  # ground truth
args.max_gen_summary_length = max_gen_summary_lengths[idx]  # fusion summary
args.no_repeat_ngram_size = no_repeat_ngram_sizes[idx]

model_names_few_shot = [
    "pegasus_xsum_train_{}_seed_{}_1".format(args.few_shot_size, args.few_shot_seed),
    "pegasus_reddit_train_{}_seed_{}_1".format(args.few_shot_size, args.few_shot_seed),
    "pegasus_samsum_train_{}_seed_{}_1".format(args.few_shot_size, args.few_shot_seed)
]
if args.few_shot:
    args.model_name = model_names_few_shot[idx]
    if args.val_dataset == "few_shot_default_val":
        args.val_dataset = "val_{}_seed_{}".format(args.few_shot_size, args.few_shot_seed)
        args.val_size = args.few_shot_size

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
    print("\nUsing device: {}".format(device))

    # tokenizer
    tokenizer = build_tokenizer(args)
    if args.encode_position and args.position_symbol != "":
        if args.full_position_encoding:
            for j in range(int(len(args.generation_methods) * args.n_candidates)):
                tokenizer.add_tokens(["{}{}".format(args.position_symbol, j)])
        else:
            for j in range(args.n_candidates):
                tokenizer.add_tokens(["{}{}".format(args.position_symbol, j)])
    if args.encode_generation_method:
        for j in range(len(args.generation_methods)):
            tokenizer.add_tokens(["GEN_{}".format(j)])

    # data
    set = args.val_dataset
    size = args.val_size
    texts, summaries, scored_summaries = load_data(set, size, args)
    print("\nLoaded new data!", len(texts), len(summaries), len(scored_summaries), len(scored_summaries[0]),
          len(scored_summaries[0][0]), len(scored_summaries[0][1]))
    texts = texts[:args.max_val_size]
    summaries = summaries[:args.max_val_size]
    scored_summaries = scored_summaries[:args.max_val_size]

    # dataset
    mode = "val"
    dataset = AbstractiveFusionDataset(mode, tokenizer, texts, summaries, scored_summaries, args)
    print("There are {} {} batches".format(int(len(dataset.texts) / args.inference_bs), set))

    # data loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.inference_bs, shuffle=False)

    # model
    pretrained_model = build_model(args)
    model = ModelAbstractiveFusion(pretrained_model, tokenizer, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nThe model has {} trainable parameters".format(n_params))
    model = model.to(device)
    if args.encode_position:
        print("Model embeddings size before", model.model.config.vocab_size, model.model.model.shared.num_embeddings)
        model.model.resize_token_embeddings(len(tokenizer))
        print("Model embeddings size after", model.model.config.vocab_size, model.model.model.shared.num_embeddings)
    if args.load_model:
        ft = ""
        if args.few_shot:
            ft = "_few_shot"
        args.load_model_path = "saved_models/{}{}/{}/pytorch_model.bin".format(
            args.dataset, ft, args.load_model_path
        )
        model.load_state_dict(torch.load(args.load_model_path))
        print("\nLOADING MODEL from: {} ".format(args.load_model_path))
    else:
        print("\nZERO-SHOT EVAL!!")

    _, texts, candidates, summaries, labels = validate(loader, tokenizer, model, args)
    print(texts[0])
    print(candidates[0])

    # evaluation
    if len(summaries) > 0:
        ### general evaluation
        new_scores, _ = overall_eval(texts, summaries, labels, args)

        ### specific evaluation

        # 1 - candidates abstractiveness recall: new words COMPARED TO THE POOL OF 1ST-STAGE CANDIDATES
        if args.evaluate_candidates_abstractiveness:
            print("\n", ">" * 20, "Evaluate - candidates abstractiveness")
            text_words, cand_words, summary_words, label_words = collect_words(texts, candidates, summaries, labels, args)
            evaluate_candidates_new_ngrams(cand_words, summary_words, args)

        # 2 - new summaries
        if args.evaluate_new_summaries:
            print("\n", ">" * 20, "Evaluate - % of fused summaries which are actually new")
            evaluate_new_summaries(candidates, summaries, args)

        # 3 - per feature
        # get base scores
        new_candidates = []
        for i in tqdm(range(len(candidates))):
            candidates_i = candidates[i]
            candidates_i = candidates_i.split(args.sep_symbol)[1:]
            if args.encode_generation_method:
                candidates_i = [" ".join(x.split()[1:]) for x in candidates_i]
            if args.encode_position:
                candidates_i = [" ".join(x.split()[1:]) for x in candidates_i]
            candidates_i = [re.sub(' +', ' ', x.lower().strip()) for x in candidates_i]
            new_candidates.append(candidates_i)
        candidates = new_candidates
        base_summaries = [candidates[i][0] for i in range(len(candidates))]
        base_scores, _ = overall_eval(texts, base_summaries, labels, args)
        # 3a - per summary quality
        if args.evaluate_per_summary_quality:
            print("\n", ">" * 20, "\nEvaluate - per summary quality:")
            summary_qualities = []
            for i in range(len(base_summaries)):
                quality_i = np.mean(np.array([base_scores[j][i] for j in range(len(base_scores))]))
                summary_qualities.append(quality_i)
            summary_qualities = np.array(summary_qualities)
            evaluate_per_splitting_feature(summary_qualities, base_scores, new_scores, args)
        # 3b - summary diversity
        if args.evaluate_per_summary_diversity:
            print("\n", ">" * 20, "\nEvaluate - per summary diversity:")
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=args.stemmer)
            summary_diversities = []
            for i in tqdm(range(len(candidates))):
                diversity_i = []
                for j in range(len(candidates[i])):
                    for k in range(j + 1, len(candidates[i])):
                        rouge_scores = scorer.score(candidates[i][j], candidates[i][k])
                        r1 = rouge_scores["rouge1"].fmeasure
                        diversity_i.append(1 - r1)
                diversity_i = np.mean(diversity_i)
                summary_diversities.append(diversity_i)
            summary_diversities = np.array(summary_diversities)
            evaluate_per_splitting_feature(summary_diversities, base_scores, new_scores, args)
        # 3c - per source length
        if args.evaluate_per_source_length:
            print("\n", ">" * 20, "\nEvaluate - per source length:")
            source_lengths = np.array([len(word_tokenize(texts[i])) for i in range(len(texts))])
            evaluate_per_splitting_feature(source_lengths, base_scores, new_scores, args)
        # 3d - per target ratio
        if args.evaluate_per_target_ratio:
            print("\n", ">" * 20, "\nEvaluate - per target ratio:")
            summary_ratios = []
            for i in range(len(candidates)):
                # lengths_i = [len(word_tokenize(candidates[i][j])) for j in range(len(candidates[i]))]
                # length_i = np.mean(np.array(lengths_i))
                length_i = len(word_tokenize(labels[i]))
                source_length = len(word_tokenize(texts[i]))
                ratio_i = length_i / max(source_length, 1)
                summary_ratios.append(ratio_i)
            summary_ratios = np.array(summary_ratios)
            evaluate_per_splitting_feature(summary_ratios, base_scores, new_scores, args)

        # 4 - break oracle barrier
        if args.evaluate_break_oracle:
            print("\n", ">" * 20, "\nEvaluate - breaking the oracle:")
            evaluate_break_oracle(candidates, summaries, labels, args)

        # 5 - ablation on the input candidates
        if args.evaluate_ablation_candidates:
            print("\n", ">" * 20, "\nEvaluate - ablation on # candidates:")
            evaluate_ablation_candidates(tokenizer, model, args)

        # 6 - ablation on the source
        if args.evaluate_without_source:
            print("\n", ">" * 20, "\nEvaluate - without source:")
            evaluate_without_source(tokenizer, model, args)



if __name__ == '__main__':
    main(args)
