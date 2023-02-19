# Complete SummaFusion generation inference pipeline in a single small script!
# Default setup in on XSum with 15 DBS candidates

import gc
import sys
sys.path.append("/data/mathieu/SummaFusion/src/")
import numpy as np
import torch
import argparse
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from transformers import PegasusTokenizerFast, PegasusForConditionalGeneration, BartTokenizerFast
from modeling_bart import BartForConditionalGeneration
from rouge_score import rouge_scorer

from common.utils import seed_everything
from model import ModelAbstractiveFusion
from engine import generation_step
from generation_utils import GenerationMixin


parser = argparse.ArgumentParser()
args = parser.parse_args()
args.device = torch.device("cuda")
args.use_source = True
args.use_candidates = True
args.classify_candidates = True
args.hidden_size = 1024
args.use_ss_for_cls = False
args.use_source_for_cls = True
args.cls_hidden_size = 2048
args.scoring_methods = ["rouge_1", "rouge_2", "rouge_l"]
args.max_candidate_length = 34
args.n_candidates_to_use = 15
args.max_source_length = 1024
args.num_gen_beams = 10
args.num_return_sequences = 1
args.max_gen_summary_length = 64
args.repetition_penalty = 1.0
args.length_penalty = 0.8
args.no_repeat_ngram_size = 3

# seed
seed_everything(50)

# data
dataset_name = "xsum"
subset = "test"
text_key = "document"
summary_key = "summary"
dataset = load_dataset(dataset_name, cache_dir="/data/mathieu/hf_datasets/")
dataset = dataset[subset]
texts = dataset[text_key]
p = np.random.permutation(len(texts))
texts = [texts[x] for x in p]
text = texts[0]
text = text.replace("\n", " ")
print("\nSource document:")
print(text)
labels = dataset[summary_key]
labels = [labels[x] for x in p]
label = labels[0]
label = "\n".join(sent_tokenize(label))
print("\nLabel:")
print(label)

# base model
base_model_name = "google/pegasus-xsum"
base_tokenizer = PegasusTokenizerFast.from_pretrained(base_model_name, cache_dir="/data/mathieu/hf_models/pegasus-large-xsum/")
base_model = PegasusForConditionalGeneration.from_pretrained(base_model_name, cache_dir="/data/mathieu/hf_models/pegasus-large-xsum/")
base_model = base_model.to(args.device)
base_model = base_model.eval()

# candidates
tok_text = base_tokenizer(text, return_tensors="pt", padding="max_length", max_length=512)
tok_text["input_ids"] = tok_text["input_ids"][:, :512]
tok_text["attention_mask"] = tok_text["attention_mask"][:, :512]
with torch.no_grad():
    generated = base_model.generate(
        input_ids=tok_text["input_ids"].to(args.device),
        attention_mask=tok_text["attention_mask"].to(args.device),
        num_beams=15,
        num_beam_groups=15,
        num_return_sequences=15,
        diversity_penalty=1.0,
        repetition_penalty=1.0,
        length_penalty=0.8,
        no_repeat_ngram_size=3
    )
candidates = base_tokenizer.batch_decode(generated, skip_special_tokens=True)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer = True)
print("\nSummary candidates:")
for j in range(len(candidates)):
    rouge_scores = scorer.score(label, candidates[j])
    r1 = 100 * rouge_scores["rouge1"].fmeasure
    r2 = 100 * rouge_scores["rouge2"].fmeasure
    rl = 100 * rouge_scores["rougeLsum"].fmeasure
    mean_r = (r1 + r2 + rl) / 3
    print("\nCandidates {} (Mean R: {:.2f}, R-1: {:.2f}, R-2: {:.2f}, R-L: {:.2f})".format(j, mean_r, r1, r2, rl))
    print(candidates[j])

del base_model
del tok_text
del generated
gc.collect()

# SummaFusion 
# model
model_name = "facebook/bart-large"
tokenizer = BartTokenizerFast.from_pretrained(model_name, cache_dir="/data/mathieu/hf_models/bart-large/")
for j in range(args.n_candidates_to_use):
    tokenizer.add_tokens(["CAND_{}".format(j)])
model = BartForConditionalGeneration.from_pretrained(model_name, args=args, cache_dir="/data/mathieu/hf_models/bart-large/")
model = model.to(args.device)
summafusion_model = ModelAbstractiveFusion(model, tokenizer, args)
summafusion_model.model.resize_token_embeddings(len(tokenizer))
summafusion_model = summafusion_model.to(args.device)
summafusion_model_path = "/data/mathieu/2nd_stage_summarization/5_abstractive_fusion/saved_models/xsum/fusion_main_5/pytorch_model.bin"
summafusion_model.load_state_dict(torch.load(summafusion_model_path))
summafusion_model = summafusion_model.eval()
# prepare the data
candidates_inputs, candidates_masks = [], []
for j in range(len(candidates)):
    candidate = candidates[j]
    candidate = "CAND_{} ".format(j) + candidate
    candidate_inputs = tokenizer(candidate, return_tensors="pt", truncation=True, padding="max_length", max_length=args.max_candidate_length)
    candidate_inputs["input_ids"] = candidate_inputs["input_ids"][:, :args.max_candidate_length]
    candidate_inputs["attention_mask"] = candidate_inputs["attention_mask"][:, :args.max_candidate_length]
    candidates_inputs.append(candidate_inputs["input_ids"][0])
    candidates_masks.append(candidate_inputs["attention_mask"][0])
candidates_inputs = torch.cat(candidates_inputs)
candidates_masks = torch.cat(candidates_masks)
source_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=args.max_source_length)
source_ids = source_inputs["input_ids"][:, :args.max_source_length]
source_mask = source_inputs["attention_mask"][:, :args.max_source_length]
cand_ids = candidates_inputs.unsqueeze(0)
cand_mask = candidates_masks.unsqueeze(0)
cand_ids = torch.cat((source_ids, cand_ids), -1)
cand_mask = torch.cat((source_mask, cand_mask), -1)
batch = {
    "cand_ids": cand_ids.to(args.device),
    "cand_mask": cand_mask.to(args.device),
    "source_ids": source_ids.to(args.device),
    "source_mask": source_mask.to(args.device)
}
# inference
gm = GenerationMixin
with torch.no_grad():
    generated = generation_step(batch, tokenizer, summafusion_model, gm, args)
summary = generated[0]
rouge_scores = scorer.score(label, summary)
r1 = 100 * rouge_scores["rouge1"].fmeasure
r2 = 100 * rouge_scores["rouge2"].fmeasure
rl = 100 * rouge_scores["rougeLsum"].fmeasure
mean_r = (r1 + r2 + rl) / 3
print("\nSummaFusion output summary (Mean R: {:.2f}, R-1: {:.2f}, R-2: {:.2f}, R-L: {:.2f})".format(mean_r, r1, r2, rl))
print(summary)
