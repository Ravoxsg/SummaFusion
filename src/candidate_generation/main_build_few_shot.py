# Subsample (train, val) subsets from the training set

import os
import argparse
import sys
import re
import numpy as np
import pickle

sys.path.append("/data/mathieu/SummaFusion/src/") # todo: change to your folder path

from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from shutil import copyfile

from common.utils import seed_everything



parser = argparse.ArgumentParser()

parser.add_argument('--seeds', type = list, default = [42, 43, 44])
parser.add_argument('--dataset', type = str, default = "samsum") # in ["cnndm", "xsum", "reddit", "samsum"]
parser.add_argument('--few_shot_sizes', type = list, default = [10, 100, 1000])
parser.add_argument('--data_folder', type = str, default = "../../data/")

args = parser.parse_args()

datasets = ["xsum", "reddit", "samsum"]
train_sizes = [204045, 33704, 14732]
val_sizes = [11332, 4213, 818]

idx = datasets.index(args.dataset)

args.train_size = train_sizes[idx]
args.val_size = val_sizes[idx]

print("*"*50)
print(args)



def main(args):
    if not(os.path.isdir("../../few_shot_permutations/")):
        os.makedirs("../../few_shot_permutations/")

    train_summaries, train_texts = load_data_txt("train", args)
    val_summaries, val_texts = load_data_txt("val", args)

    for k in range(len(args.seeds)):
        seed = args.seeds[k]
        seed_everything(seed)

        p = np.random.permutation(len(train_summaries))
        with open("../../few_shot_permutations/{}_seed_{}_train_permutation.pkl".format(args.dataset, seed), "wb") as f:
            pickle.dump(p, f)
            print("saved training permutation!")
        train_summaries = [train_summaries[x] for x in p]
        train_texts = [train_texts[x] for x in p]

        p = np.random.permutation(len(val_summaries))
        with open("../../few_shot_permutations/{}_seed_{}_val_permutation.pkl".format(args.dataset, seed), "wb") as f:
            pickle.dump(p, f)
            print("saved val permutation!")

        val_summaries = [val_summaries[x] for x in p]
        val_texts = [val_texts[x] for x in p]
        path = args.data_folder + "{}/".format(args.dataset)
        for i in range(len(args.few_shot_sizes)):
            size = args.few_shot_sizes[i]
            print("*"*50)
            print("Building few-shot size: {}".format(size))
            few_train_summaries = train_summaries[:size]
            few_train_texts = train_texts[:size]
            few_val_summaries = val_summaries[:size]
            few_val_texts = val_texts[:size]
            with open(path + "train_{}_seed_{}_summary.txt".format(size, seed), "w") as f:
                for l in few_train_summaries:
                    f.write(l)
            with open(path + "train_{}_seed_{}_text.txt".format(size, seed), "w") as f:
                for l in few_train_texts:
                    f.write(l)
            print("wrote the training files!")

            with open(path + "val_{}_seed_{}_summary.txt".format(size, seed), "w") as f:
                for l in few_val_summaries:
                    f.write(l)
            with open(path + "val_{}_seed_{}_text.txt".format(size, seed), "w") as f:
                for l in few_val_texts:
                    f.write(l)
            print("wrote the validation files!")


def load_data_txt(set, args):
    summaries, texts, top_sents = [], [], []
    path = args.data_folder + "{}/".format(args.dataset)
    with open(path + "{}_summary.txt".format(set), "r") as f:
        for l in f.readlines():
            summaries.append(l)
    with open(path + "{}_text.txt".format(set), "r") as f:
        for l in f.readlines():
            texts.append(l)
    print(len(summaries), len(texts),)

    return summaries, texts



if __name__ == '__main__':
    main(args)




