import sys
sys.path.append("/data/mathieu/SummaFusion/src/")
import numpy as np
import torch
import argparse
from datasets import load_dataset
from transformers import PegasusTokenizerFast, PegasusForConditionalGeneration, BartTokenizerFast
from modeling_bart import BartForConditionalGeneration

from common.utils import seed_everything
from model import ModelAbstractiveFusion
from engine import generation_step
from generation_utils import GenerationMixin


# seed
seed_everything(42)

# data
dataset_name = "xsum"
subset = "test"
text_key = "document"
dataset = load_dataset(dataset_name, cache_dir="data/mathieu/hf_datasets/")
dataset = dataset[subset]
texts = dataset[text_key]
p = np.random.permutation(len(texts))
texts = [texts[x] for x in p]
text = texts[0]
text = text.replace("\n", " ")
print("\nSource document:")
print(text)

# base model
base_model_name = "google/pegasus-xsum"
base_tokenizer = PegasusTokenizerFast.from_pretrained(base_model_name, cache_dir="/data/mathieu/hf_models/pegasus-large-xsum/")
base_model = PegasusForConditionalGeneration.from_pretrained(base_model_name, cache_dir="/data/mathieu/hf_models/pegasus-large-xsum/")
base_model = base_model.cuda()

# candidates
tok_text = base_tokenizer(text, return_tensors="pt", padding="max_length", max_length=512)
tok_text["input_ids"] = tok_text["input_ids"][:, :512]
tok_text["attention_mask"] = tok_text["attention_mask"][:, :512]
generated = base_model.generate(
    input_ids=tok_text["input_ids"].cuda(),
    attention_mask=tok_text["attention_mask"].cuda(),
    num_beams=15,
    num_beam_groups=15,
    num_return_sequences=15,
    diversity_penalty=1.0,
    repetition_penalty=1.0,
    length_penalty=0.8,
    no_repeat_ngram_size=3
)
candidates = base_tokenizer.batch_decode(generated, skip_special_tokens=True)
print("\nSummary candidates:")
for j in range(len(candidates)):
    print("Candidate {}:".format(j))
    print(candidates[j])

# SummaFusion 
# model
model_name = "facebook/bart-large"
tokenizer = BartTokenizerFast.from_pretrained(model_name, cache_dir="/data/mathieu/hf_models/bart-large/")
model = BartForConditionalGeneration.from_pretrained(model_name, args=None, cache_dir="/data/mathieu/hf_models/bart-large/")
model = model.cuda()
summafusion_model = ModelAbstractiveFusion(model, tokenizer, None)
summafusion_model = summafusion_model.cuda()
# prepare the data
max_candidate_length = 34
candidates_inputs, candidates_masks = [], []
for j in range(len(candidates)):
    candidate = candidates[j]
    candidate = "CAND_{} ".format(j) + candidate
    candidate_inputs = tokenizer(candidate, return_tensors="pt", truncation=True, padding="max_length", max_length=max_candidate_length)
    candidate_inputs["input_ids"] = candidate_inputs["input_ids"][:, :max_candidate_length]
    candidate_inputs["attention_mask"] = candidate_inputs["attention_mask"][:, :max_candidate_length]
    candidates_inputs.append(candidate_inputs["input_ids"][0])
    candidates_masks.append(candidate_inputs["attention_mask"][0])
candidates_inputs = torch.cat(candidates_inputs)
candidates_masks = torch.cat(candidates_masks)
print(candidates_inputs.shape, candidates_masks.shape)
source_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
source_ids = source_inputs["input_ids"][:, :1024]
source_mask = source_inputs["attention_mask"][:, :1024]
batch = {
        "cand_ids": candidates_inputs.unsqueeze(0),
        "cand_mask": candidates_masks.unsqueeze(0),
        "source_ids": source_ids,
        "source_mask": source_mask
}
# inference
gm = GenerationMixin
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.device = torch.device("cuda")
args.max_source_length = 1024
args.use_source = True
args.use_candidates = True
args.num_gen_beams = 10
args.num_return_sequences = 1
args.max_gen_summary_length = 64
args.repetition_penalty = 1.0
args.length_penalty = 0.8
args.no_repeat_ngram_size = 3
generated = generation_step(batch, tokenizer, summafusion_model, gm, args)
print(generated)



