import os

from tqdm import tqdm


def load_data(set, args):
    text_files, summary_files = prepare_data_files(set, args)
    texts, summaries = read_data_files(text_files, summary_files)

    print("Total # of texts: {}".format(len(texts)))

    return texts, summaries


def prepare_data_files(set, args):
    # find the files
    text_files = []
    summary_files = []
    text_file =  "../../data/{}/{}_text.txt".format(args.dataset, set)
    summary_file = "../../data/{}/{}_summary.txt".format(args.dataset, set)
    text_files.append(text_file)
    summary_files.append(summary_file)

    print("For set {}, loading the following files:".format(set))
    print(text_files)
    print(summary_files)

    return text_files, summary_files


def read_data_files(text_files, summary_files):
    # read the .txt files
    texts = []
    summaries = []

    for text_file in text_files:
        lines = read_one_file(text_file)
        texts += lines
    for summary_file in summary_files:
        lines = read_one_file(summary_file)
        summaries += lines

    return texts, summaries


def read_one_file(file):
    lines = []
    with open(file, 'r') as f:
        for l in tqdm(f.readlines()):
            lines.append(l)
    print(file, len(lines))

    return lines
