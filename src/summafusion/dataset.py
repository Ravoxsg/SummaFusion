import torch
import numpy as np

from time import time



class AbstractiveFusionDataset:
    def __init__(self, mode, tokenizer, texts, labels, scored_summaries, args):
        self.mode = mode
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.scored_summaries = scored_summaries
        self.args = args

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        source = self.texts[item]
        scored_summaries = self.scored_summaries[item]
        candidates = scored_summaries[0]
        scores = scored_summaries[1]
        label = self.labels[item]

        mode = torch.tensor([1])
        if self.mode != "train":
            mode = torch.tensor([0])

        # source
        if torch.sum(mode) > 0 or self.args.source_dropout_at_inference:
            if self.args.source_dropout:
                p = np.random.uniform(0, 1)
                if p <= self.args.source_dropout_prob:
                    source = "MASK0"
        source_inputs = self.tokenizer(source, return_tensors = "pt", truncation = True, max_length = self.args.max_source_length, padding = 'max_length')
        source_inputs["input_ids"] = source_inputs["input_ids"][:, :self.args.max_source_length]
        source_inputs["attention_mask"] = source_inputs["attention_mask"][:, :self.args.max_source_length]

        # candidates 
        # candidates_no_ss, candidates, all_candidates, all_candidate_inputs, all_candidate_masks, scores = self.build_candidates(candidates, scores, mode)
        all_candidates, all_candidates_block, all_candidates_inputs, all_candidates_masks, candidates, candidates_block, candidates_inputs, candidates_masks, scores = self.build_candidates(candidates, scores, mode)
        candidates_inputs = torch.cat((candidates_inputs, all_candidates_inputs), 0)
        candidates_masks = torch.cat((candidates_masks, all_candidates_masks), 0)

        # labels
        label_inputs = self.tokenizer(label, return_tensors = "pt", truncation = True, max_length = self.args.max_summary_length, padding = 'max_length')
        label_inputs["input_ids"] = label_inputs["input_ids"][:, :self.args.max_summary_length]
        label_inputs["attention_mask"] = label_inputs["attention_mask"][:, :self.args.max_summary_length]

        # classification labels
        candidates_to_cls = all_candidates
        if self.args.use_ss_for_cls:
            candidates_to_cls = candidates
        cls_labels = self.build_cls_labels(mode, candidates_to_cls, scores)

        batch = {
            "mode": mode,

            "source": source,
            "source_ids": source_inputs["input_ids"][0],
            "source_mask": source_inputs["attention_mask"][0],

            "candidates": candidates_block,
            "cand_ids": candidates_inputs,
            "cand_mask": candidates_masks,

            "label": label,
            "label_ids": label_inputs["input_ids"][0],
            "label_mask": label_inputs["attention_mask"][0],

            "cls_labels": cls_labels
        }

        return batch

    def build_candidates(self, candidates, scores, mode):
        # ordering
        candidates, scores = self.candidates_ordering(candidates, scores)

        # encode position
        if self.args.encode_position:
            if self.args.position_symbol != "" and not(candidates[0].startswith(self.args.position_symbol)):
                for i in range(len(candidates)):
                    if self.args.full_position_encoding:
                        num = i
                    else:
                        num = i % self.args.num_beams
                    candidates[i] = self.args.position_symbol + "{} ".format(num) + candidates[i]
            if self.args.position_symbol == "" and not((len(candidates[0]) > 0) and candidates[0][0].isnumeric()):
                for i in range(len(candidates)):
                    if self.args.full_position_encoding:
                        num = i
                    else:
                        num = i % self.args.num_beams
                    candidates[i] = "{} ".format(num) + candidates[i]
        # encode generation method
        if self.args.encode_generation_method:
            for i in range(len(candidates)):
                gen = int(i / self.args.n_candidates)
                candidates[i] = "GEN_{} ".format(gen) + candidates[i]

        # subsetting
        candidates = candidates[:self.args.n_candidates_to_use]
        scores = scores[:self.args.n_candidates_to_use]

        all_candidates = candidates

        # subsampling
        if torch.sum(mode) > 0 or self.args.subsample_at_inference:
            candidates = self.candidates_subsampling(candidates, scores)

        # shuffle candidates
        if self.args.shuffle_candidates:
            p = np.random.permutation(len(candidates))
            all_candidates = [all_candidates[x] for x in p]
            candidates = [candidates[x] for x in p]
            scores = [scores[x] for x in p]

        # tokenizing - all candidates
        all_candidates_block = ""
        all_candidates_inputs, all_candidates_masks = [], []
        for i in range(len(all_candidates)):
            candidate = all_candidates[i]
            candidate_inputs = self.tokenizer(candidate, return_tensors = "pt", truncation = True, max_length = self.args.max_candidate_length, padding = 'max_length')
            if i == 0:
                all_candidates_block += self.args.sep_symbol + " " + self.tokenizer.decode(candidate_inputs["input_ids"][0], skip_special_tokens = True)
            else:
                all_candidates_block += " " + self.args.sep_symbol + " " + self.tokenizer.decode(candidate_inputs["input_ids"][0], skip_special_tokens = True)
            candidate_inputs["input_ids"] = candidate_inputs["input_ids"][:, :self.args.max_candidate_length]
            candidate_inputs["attention_mask"] = candidate_inputs["attention_mask"][:, :self.args.max_candidate_length]
            all_candidates_inputs.append(candidate_inputs["input_ids"][0])
            all_candidates_masks.append(candidate_inputs["attention_mask"][0])
        all_candidates_inputs = torch.cat(all_candidates_inputs)
        all_candidates_masks = torch.cat(all_candidates_masks)

        # tokenizing - candidates
        candidates_block = ""
        candidates_inputs, candidates_masks = [], []
        for i in range(len(candidates)):
            candidate = candidates[i]
            candidate_inputs = self.tokenizer(candidate, return_tensors = "pt", truncation = True, max_length = self.args.max_candidate_length, padding = 'max_length')
            if i == 0:
                candidates_block += self.args.sep_symbol + " " + self.tokenizer.decode(candidate_inputs["input_ids"][0], skip_special_tokens = True)
            else:
                candidates_block += " " + self.args.sep_symbol + " " + self.tokenizer.decode(candidate_inputs["input_ids"][0], skip_special_tokens = True)
            candidate_inputs["input_ids"] = candidate_inputs["input_ids"][:, :self.args.max_candidate_length]
            candidate_inputs["attention_mask"] = candidate_inputs["attention_mask"][:, :self.args.max_candidate_length]
            candidates_inputs.append(candidate_inputs["input_ids"][0])
            candidates_masks.append(candidate_inputs["attention_mask"][0])
        candidates_inputs = torch.cat(candidates_inputs)
        candidates_masks = torch.cat(candidates_masks)

        return all_candidates, all_candidates_block, all_candidates_inputs, all_candidates_masks, candidates, candidates_block, candidates_inputs, candidates_masks, scores

    def candidates_ordering(self, candidates, scores):
        idx = list(range(len(candidates)))
        candidates = [candidates[i] for i in idx]
        new_scores = []
        for j in range(len(scores)):
            scores_j = [scores[j][i] for i in idx]
            new_scores.append(scores_j)
        scores = []
        for i in range(len(new_scores[0])):
            scores_i = [new_scores[j][i] for j in range(len(new_scores))]
            scores.append(scores_i)

        return candidates, scores

    def candidates_subsampling(self, candidates, scores):
        idx = list(range(len(candidates)))
        if self.args.cand_subsampling_method != "":
            n_to_subsample = np.random.randint(self.args.n_subsample_low, self.args.n_subsample_high + 1)
            if self.args.cand_subsampling_method == "random":
                p = np.random.permutation(len(candidates))
                idx = p[:n_to_subsample]
                idx.sort()
            elif self.args.cand_subsampling_method == "top":
                idx = np.argsort(scores)[::-1]
                idx = idx[:n_to_subsample]
                idx.sort()
        new_candidates = []
        for i in range(len(candidates)):
            if i in idx:
                new_candidates.append(candidates[i])
            else:
                new_candidates.append("MASK1")

        return new_candidates

    def build_cls_labels(self, mode, candidates, scores):
        scores = torch.tensor(scores)
        cls_labels = -1 * torch.ones(scores.shape)
        idx = np.array(candidates) != "MASK1"
        for j in range(scores.shape[1]):
            scores_j = scores[idx, j]
            idx_sort_idx = torch.argsort(scores_j, descending = True)
            sort_idx = np.arange(len(candidates))[idx][idx_sort_idx]
            # positives
            max_j = torch.max(scores_j)
            idx_pos_idx = np.arange(len(scores_j))[scores_j[idx_sort_idx] == max_j]
            pos_thresh = len(idx_pos_idx)
            if pos_thresh == len(idx_sort_idx) or (torch.sum(mode) > 0 and self.args.subsample_cls_cands):
                pos_thresh = 1
            pos_idx = sort_idx[:pos_thresh]
            cls_labels[pos_idx, j] = 1
            # negatives
            neg_thresh = len(sort_idx) - pos_thresh
            if torch.sum(mode) > 0 and self.args.subsample_cls_cands:
                neg_thresh = self.args.n_subsample_cls_neg
            neg_idx = sort_idx[-neg_thresh:]
            cls_labels[neg_idx, j] = 0

        return cls_labels




