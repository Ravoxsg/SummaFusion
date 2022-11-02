# Separate the training sets into 2 parts

import argparse
import sys
import numpy as np
import pickle

sys.path.append("/data/mathieu/2nd_stage_summarization/")

from tqdm import tqdm
from shutil import copyfile

from common.utils import seed_everything



parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--dataset', type=str, default = "samsum",
                    choices=["xsum", "reddit", "samsum"])
parser.add_argument('--few_shot', type = bool, default = True)
parser.add_argument('--training_set', type = str, default = "train_100_seed_42",
                    choices = [
                        "train",
                        "train_10_seed_42", "train_10_seed_43", "train_10_seed_44",
                        "train_100_seed_42", "train_100_seed_43", "train_100_seed_44",
                        "train_1000_seed_42", "train_1000_seed_43", "train_1000_seed_44",
                    ])

args = parser.parse_args()

dataset_names = ["xsum", "reddit", "samsum"]
threshs = [102000, 17000, 7350]

idx = dataset_names.index(args.dataset)
args.data_folder = "../../data/{}/".format(args.dataset)
args.thresh = threshs[idx]
if args.few_shot:
    args.few_shot_size = int(args.training_set.split("_")[1])
    args.thresh = int(args.few_shot_size/2)

print("*"*50)
print(args)



def main(args):
    # seed
    seed_everything(args.seed)

    # load full training
    train_summaries, train_texts = [], []
    with open(args.data_folder + "train_summary.txt", "rb") as f:
        for l in f.readlines():
            train_summaries.append(l)
    with open(args.data_folder + "train_text.txt", "rb") as f:
        for l in f.readlines():
            train_texts.append(l)
    print(len(train_summaries), len(train_texts))

    # shuffle
    p = np.random.permutation(len(train_texts))
    print(p[:10])
    with open("dataset_permutations/{}_train_permutation.pkl".format(args.dataset), "wb") as f:
        pickle.dump(p, f)
        print("saved permutation!")
    train_summaries = [train_summaries[i] for i in p]
    train_texts = [train_texts[i] for i in p]
    print("permuted the training set!")
    
    if args.few_shot:
        train_summaries = train_summaries[:args.few_shot_size]
        train_texts = train_texts[:args.few_shot_size]
        print(len(train_texts))

    # 1st half - full files
    first_half_summaries = train_summaries[:args.thresh]
    first_half_texts = train_texts[:args.thresh]
    print(len(first_half_summaries), len(first_half_texts))
    with open(args.data_folder + "first_half_{}_shuffled_summary.txt".format(args.training_set), "wb") as f:
        for l in first_half_summaries:
            f.write(l)
    with open(args.data_folder + "first_half_{}_shuffled_text.txt".format(args.training_set), "wb") as f:
        for l in first_half_texts:
            f.write(l)

    # 2nd half - full files
    second_half_summaries = train_summaries[args.thresh:]
    second_half_texts = train_texts[args.thresh:]
    print(len(second_half_summaries), len(second_half_texts))
    with open(args.data_folder + "second_half_{}_shuffled_summary.txt".format(args.training_set), "wb") as f:
        for l in second_half_summaries:
            f.write(l)
    with open(args.data_folder + "second_half_{}_shuffled_text.txt".format(args.training_set), "wb") as f:
        for l in second_half_texts:
            f.write(l)
            


if __name__ == '__main__':
    main(args)




