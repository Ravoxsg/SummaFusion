import argparse
import sys
import numpy as np 
import torch
import pickle
import scipy.special
import operator
import re
import json
import torch.nn as nn
sys.path.append("/data/mathieu/SummaFusion/src/")
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from difflib import SequenceMatcher
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

from common.evaluation import *
from common.data_scored import load_data
from dataset import AbstractiveFusionDataset
from training_utils import *
from engine import validate


# for 1
def collect_words(texts, candidates, summaries, labels, args):
    all_text_words, all_cand_words, all_summary_words, all_label_words = [], [], [], []
    for i in tqdm(range(len(texts))):
        text = texts[i]
        candidates_i = candidates[i]
        summary = summaries[i]
        label = labels[i]        

        text_words = text.lower()
        #text_words = text_words.split()
        text_words = word_tokenize(text_words)
        all_text_words.append(text_words)

        cand_words = candidates_i.replace(args.sep_symbol, " ").lower()
        remove_tokens = []
        if args.encode_generation_method:
            remove_tokens += ["gen_{}".format(j) for j in range(len(args.generation_methods))]
        if args.encode_position:
            remove_tokens += ["cand_{}".format(j) for j in range(int(len(args.scoring_methods) * args.num_beams))]
        cand_words = " ".join([x for x in cand_words.split() if not(x in remove_tokens)])
        #cand_words = cand_words.split()
        cand_words = word_tokenize(cand_words)
        all_cand_words.append(cand_words)

        summary_words = summary.lower()
        #summary_words = summary_words.split()
        summary_words = word_tokenize(summary_words)
        all_summary_words.append(summary_words)

        label_words = label.lower()
        #label_words = label_words.split()
        label_words = word_tokenize(label_words)
        all_label_words.append(label_words)

    return all_text_words, all_cand_words, all_summary_words, all_label_words


# 1
def evaluate_candidates_new_ngrams(cand_words, summary_words, args):
    new_unigrams_counts = {}
    new_unigrams, new_bigrams, new_trigrams, new_quadrigrams = [], [], [], []
    for i in tqdm(range(len(cand_words))):
        # candidates
        cand_bigrams = [[cand_words[i][j], cand_words[i][j + 1]] for j in range(len(cand_words[i]) - 1)]
        cand_trigrams = [[cand_words[i][j], cand_words[i][j + 1], cand_words[i][j + 2]] for j in
                         range(len(cand_words[i]) - 2)]
        cand_quadrigrams = [[cand_words[i][j], cand_words[i][j + 1], cand_words[i][j + 2], cand_words[i][j + 3]] for j
                            in range(len(cand_words[i]) - 3)]

        unigrams, bigrams, trigrams, quadrigrams = 0, 0, 0, 0
        for j in range(len(summary_words[i])):
            if not (summary_words[i][j] in cand_words[i]):
                unigrams += 1
                if not (summary_words[i][j] in new_unigrams_counts.keys()):
                    new_unigrams_counts[summary_words[i][j]] = 0
                new_unigrams_counts[summary_words[i][j]] += 1
            if j < len(summary_words[i]) - 1:
                bigram = [summary_words[i][j], summary_words[i][j + 1]]
                if not (bigram in cand_bigrams):
                    bigrams += 1
            if j < len(summary_words[i]) - 2:
                trigram = [summary_words[i][j], summary_words[i][j + 1], summary_words[i][j + 2]]
                if not (trigram in cand_trigrams):
                    trigrams += 1
            if j < len(summary_words[i]) - 3:
                quadrigram = [summary_words[i][j], summary_words[i][j + 1], summary_words[i][j + 2],
                              summary_words[i][j + 3]]
                if not (quadrigram in cand_quadrigrams):
                    quadrigrams += 1
        if len(summary_words[i]) > 0:
            new_unigrams.append(unigrams / len(summary_words[i]))
        if len(summary_words[i]) > 1:
            new_bigrams.append(bigrams / (len(summary_words[i]) - 1))
        if len(summary_words[i]) > 2:
            new_trigrams.append(trigrams / (len(summary_words[i]) - 2))
        if len(summary_words[i]) > 3:
            new_quadrigrams.append(quadrigrams / (len(summary_words[i]) - 3))
    new_unigrams = np.array(new_unigrams)
    m_uni = 100 * np.mean(new_unigrams)
    new_bigrams = np.array(new_bigrams)
    m_bi = 100 * np.mean(new_bigrams)
    new_trigrams = np.array(new_trigrams)
    m_tri = 100 * np.mean(new_trigrams)
    new_quadrigrams = np.array(new_quadrigrams)
    m_quadri = 100 * np.mean(new_quadrigrams)
    print("\nNew (not in CANDIDATES) unigrams: {:.2f}, bigrams: {:.2f}, trigrams: {:.2f}, quadrigrams: {:.2f}".format(
        m_uni, m_bi, m_tri, m_quadri))
    sorted_d = dict(sorted(new_unigrams_counts.items(), key=operator.itemgetter(1), reverse=True))
    n_new = sum([sorted_d[k] for k in sorted_d.keys()])
    for idx, k in enumerate(sorted_d.keys()):
        print(
            "Most common new word (not in CANDIDATES) # {} is ** {} **, count: {} ({:.4f}%)".format(idx, k, sorted_d[k],
                                                                                                    100 * sorted_d[
                                                                                                        k] / n_new))
        if idx >= 2:
            break


def evaluate_source_and_candidates_new_ngrams(text_words, cand_words, summary_words, args):
    new_unigrams_counts = {}
    new_unigrams, new_bigrams, new_trigrams, new_quadrigrams = [], [], [], []
    for i in tqdm(range(len(text_words))):
        # text
        text_bigrams = [[text_words[i][j], text_words[i][j + 1]] for j in range(len(text_words[i]) - 1)]
        text_trigrams = [[text_words[i][j], text_words[i][j + 1], text_words[i][j + 2]] for j in range(len(text_words[i]) - 2)]
        text_quadrigrams = [[text_words[i][j], text_words[i][j + 1], text_words[i][j + 2], text_words[i][j + 3]] for j in range(len(text_words[i]) - 3)]
        
        # candidates
        cand_bigrams = [[cand_words[i][j], cand_words[i][j + 1]] for j in range(len(cand_words[i]) - 1)]
        cand_trigrams = [[cand_words[i][j], cand_words[i][j + 1], cand_words[i][j + 2]] for j in range(len(cand_words[i]) - 2)]
        cand_quadrigrams = [[cand_words[i][j], cand_words[i][j + 1], cand_words[i][j + 2], cand_words[i][j + 3]] for j in range(len(cand_words[i]) - 3)]
        
        unigrams, bigrams, trigrams, quadrigrams = 0, 0, 0, 0
        for j in range(len(summary_words[i])):
            if not(summary_words[i][j] in cand_words[i]) and not(summary_words[i][j] in text_words[i]):
                unigrams += 1
                if not(summary_words[i][j] in new_unigrams_counts.keys()):
                    new_unigrams_counts[summary_words[i][j]] = 0
                new_unigrams_counts[summary_words[i][j]] += 1
            if j < len(summary_words[i]) - 1:
                bigram = [summary_words[i][j], summary_words[i][j + 1]]
                if not(bigram in cand_bigrams) and not(bigram in text_bigrams):
                    bigrams += 1
            if j < len(summary_words[i]) - 2:
                trigram = [summary_words[i][j], summary_words[i][j + 1], summary_words[i][j + 2]]
                if not(trigram in cand_trigrams) and not(trigram in text_trigrams):
                    trigrams += 1
            if j < len(summary_words[i]) - 3:
                quadrigram = [summary_words[i][j], summary_words[i][j + 1], summary_words[i][j + 2], summary_words[i][j + 3]]
                if not(quadrigram in cand_quadrigrams) and not(quadrigram in text_quadrigrams):
                    quadrigrams += 1
        if len(summary_words[i]) > 0:
            new_unigrams.append(unigrams / len(summary_words[i]))
        if len(summary_words[i]) > 1:
            new_bigrams.append(bigrams / (len(summary_words[i]) - 1))
        if len(summary_words[i]) > 2:
            new_trigrams.append(trigrams / (len(summary_words[i]) - 2))
        if len(summary_words[i]) > 3:
            new_quadrigrams.append(quadrigrams / (len(summary_words[i]) - 3))
    new_unigrams = np.array(new_unigrams)
    m_uni = 100 * np.mean(new_unigrams)
    new_bigrams = np.array(new_bigrams)
    m_bi = 100 * np.mean(new_bigrams)
    new_trigrams = np.array(new_trigrams)
    m_tri = 100 * np.mean(new_trigrams)
    new_quadrigrams = np.array(new_quadrigrams)
    m_quadri = 100 * np.mean(new_quadrigrams)
    print("\nNew (not in CANDIDATES NOR SOURCE) unigrams: {:.2f}, bigrams: {:.2f}, trigrams: {:.2f}, quadrigrams: {:.2f}".format(m_uni, m_bi, m_tri, m_quadri))
    sorted_d = dict(sorted(new_unigrams_counts.items(), key=operator.itemgetter(1),reverse=True))
    n_new = sum([sorted_d[k] for k in sorted_d.keys()])
    for idx, k in enumerate(sorted_d.keys()):
        print("Most common new word (not in CANDIDATES NOR SOURCE) # {} is ** {} **, count: {} ({:.4f}%)".format(idx, k, sorted_d[k], 100 * sorted_d[k] / n_new))
        if idx >= 2: 
            break


# 2
def evaluate_new_summaries(candidates, summaries, args):
    n_new = 0
    idx_count = {}
    for j in range(args.n_candidates_to_use):
        idx_count[j] = 0
    for i in tqdm(range(len(summaries))):
        clean_summary = re.sub(' +', ' ', summaries[i].lower().strip())
        candidates_i = candidates[i].split(args.sep_symbol)[1:]
        if args.encode_generation_method:
            candidates_i = [" ".join(x.split()[1:]) for x in candidates_i]
        if args.encode_position:
            candidates_i = [" ".join(x.split()[1:]) for x in candidates_i]
        candidates_i = [re.sub(' +', ' ', x.lower().strip()) for x in candidates_i]
        if not(clean_summary in candidates_i):
            n_new += 1
        else:
            for j in range(len(candidates_i)):
                if candidates_i[j] == clean_summary:
                    idx_count[j] += 1
    new_frac = 100 * n_new / len(summaries)
    print("\nFused summaries not among candidates: {:.4f}".format(new_frac))
    n_not_new = sum([idx_count[k] for k in idx_count.keys()])
    if args.show_distribution_over_candidates:
        for k in idx_count.keys():
            print("# times fused summary is equal to candidates {}: {} ({:.4f}%)".format(k, idx_count[k], 100 * idx_count[k] / n_not_new))



# 3
def evaluate_per_splitting_feature(splitting_feature, base_scores, new_scores, args):
    width = int(100 / args.n_bins)
    lows, bins_feature, bins_base, bins_new, bins_gain_abs, bins_gain_rel, counts, counts_rel_gain = [], [], [], [], [], [], [], []
    for i in range(args.n_bins):
        low_perc = i * width
        low = np.percentile(splitting_feature, low_perc)
        lows.append(low)
        high_perc = (i+1) * width
        high = np.percentile(splitting_feature, high_perc)
        idx = (splitting_feature >= low) * (splitting_feature < high)
        idx = np.arange(len(splitting_feature))[idx]
        feat_i = splitting_feature[idx]
        mean_feat_i = np.mean(feat_i)
        bins_feature.append(mean_feat_i)
        base_scores_i = [[base_scores[k][j] for j in idx] for k in range(len(base_scores))]
        mean_base_i = np.mean(np.array([np.array(base_scores_i[k]) for k in range(len(base_scores_i))]))
        new_scores_i = [[new_scores[k][j] for j in idx] for k in range(len(new_scores))]
        mean_new_i = np.mean(np.array([np.array(new_scores_i[k]) for k in range(len(new_scores_i))]))
        gains_i_abs, gains_i_rel = [], []
        for j in range(len(idx)):
            mean_r_base_i = np.mean(np.array([base_scores_i[k][j] for k in range(len(base_scores_i))]))
            mean_r_new_i = np.mean(np.array([new_scores_i[k][j] for k in range(len(new_scores_i))]))
            gain_j_abs = mean_r_new_i - mean_r_base_i
            gains_i_abs.append(gain_j_abs)
            if mean_r_base_i > 0:
                gain_j_rel = 100 * (mean_r_new_i - mean_r_base_i) / mean_r_base_i
                gains_i_rel.append(gain_j_rel)
        gains_i_abs = np.array(gains_i_abs)
        mean_gain_i_abs = np.mean(gains_i_abs)
        print(i, low_perc, low, high_perc, high, mean_gain_i_abs)
        gains_i_rel = np.array(gains_i_rel)
        mean_gain_i_rel = np.mean(gains_i_rel)
        bins_base.append(mean_base_i)
        bins_new.append(mean_new_i)
        bins_gain_abs.append(mean_gain_i_abs)
        bins_gain_rel.append(mean_gain_i_rel)
        counts.append(len(gains_i_abs))
        counts_rel_gain.append(len(gains_i_rel))
    lows = ["{:.2f}".format(x) for x in lows]
    bins_feat = ["{:.4f}".format(x) for x in bins_feature]
    bins_base = ["{:.4f}".format(x) for x in bins_base]
    bins_new = ["{:.4f}".format(x) for x in bins_new]
    bins_gain_abs = ["{:.4f}".format(x) for x in bins_gain_abs]
    bins_gain_rel = ["{:.4f} ({}/{})".format(bins_gain_rel[i], counts_rel_gain[i], counts[i]) for i in range(len(bins_gain_rel))]
    print("Feature low thresholds: {}".format(lows))
    print("Mean feature value across bins: {}".format(bins_feat))
    print("Base scores across bins: {}".format(bins_base))
    print("New scores across bins: {}".format(bins_new))
    print("Absolute gains across bins: {}".format(bins_gain_abs))
    print("Relative gains across bins: {}".format(bins_gain_rel))


# 4
def evaluate_break_oracle(candidates, summaries, labels, args):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=args.stemmer)
    all_oracles, all_sfs = [], []
    for i in tqdm(range(len(candidates))):
        # score base candidates
        mean_rs = []
        for j in range(len(candidates[i])):
            candidate = candidates[i][j]
            rouge_scores = scorer.score(labels[i], candidate)
            r1 = 100 * rouge_scores["rouge1"].fmeasure
            r2 = 100 * rouge_scores["rouge2"].fmeasure
            rl = 100 * rouge_scores["rougeLsum"].fmeasure
            mean_r = (r1 + r2 + rl) / 3
            mean_rs.append(mean_r)
        mean_rs = np.array(mean_rs)
        oracle_mean_r = np.max(mean_rs)
        all_oracles.append(oracle_mean_r)
        # score summafusion
        rouge_scores = scorer.score(labels[i], summaries[i])
        r1 = 100 * rouge_scores["rouge1"].fmeasure
        r2 = 100 * rouge_scores["rouge2"].fmeasure
        rl = 100 * rouge_scores["rougeLsum"].fmeasure
        mean_r = (r1 + r2 + rl) / 3
        all_sfs.append(mean_r)
    all_oracles = np.array(all_oracles)
    all_sfs = np.array(all_sfs)
    print("Mean R - oracle: {:.4f}".format(np.mean(all_oracles)))
    print("Mean R - SF: {:.4f}".format(np.mean(all_sfs)))
    sf_greater = np.sum(all_sfs > all_oracles)
    print("SF beating oracle: {} or {:.4f}%".format(sf_greater, 100 * sf_greater / len(all_sfs)))


# 5
def evaluate_ablation_candidates(tokenizer, model, args):
    args.subsample_at_inference = True
    args.cand_subsampling_method = "random"
    texts, labels, scored_summaries = load_data(args.val_dataset, args.val_size, args, individual_txt = args.highlights)
    texts = texts[:args.max_val_size]
    labels = labels[:args.max_val_size]
    scored_summaries = scored_summaries[:args.max_val_size]
    all_r1s, all_r2s, all_rls = [], [], []
    for n_cands in args.n_ablation_candidates:
        print("\nInference with {} candidates".format(n_cands))
        args.n_subsample_low = n_cands
        args.n_subsample_high = n_cands
        dataset = AbstractiveFusionDataset("val", tokenizer, texts, labels, scored_summaries, args)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.inference_bs, shuffle=False)
        _, val_texts, val_candidates, val_summaries, val_labels = validate(loader, tokenizer, model, args)
        scores, _ = overall_eval(val_texts, val_summaries, val_labels, args)
        r1s = scores[0]
        r2s = scores[1]
        rls = scores[2]
        m_r1 = np.mean(r1s)
        m_r2 = np.mean(r2s)
        m_rl = np.mean(rls)
        all_r1s.append(m_r1)
        all_r2s.append(m_r2)
        all_rls.append(m_rl)
    print("\n" + ">"*20)
    for i in range(len(args.n_ablation_candidates)):
        print("# Candidates = {}, R-1: {:.4f}, R-2: {:.4f}, R-L: {:.4f}".format(
            args.n_ablation_candidates[i], all_r1s[i], all_r2s[i], all_rls[i]))


# 6
def evaluate_without_source(tokenizer, model, args):
    args.source_dropout = True
    args.source_dropout_at_inference = True
    args.source_dropout_prob = 1.0
    texts, labels, scored_summaries = load_data(args.val_dataset, args.val_size, args, individual_txt = args.highlights)
    texts = texts[:args.max_val_size]
    labels = labels[:args.max_val_size]
    scored_summaries = scored_summaries[:args.max_val_size]
    dataset = AbstractiveFusionDataset("val", tokenizer, texts, labels, scored_summaries, args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.inference_bs, shuffle=False)
    print("\nInference with no source")
    _, val_texts, val_candidates, val_summaries, val_labels = validate(loader, tokenizer, model, args)
    scores, _ = overall_eval(val_texts, val_summaries, val_labels, args)

