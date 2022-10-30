# Generate summary candidates with the fine-tuned models.

import time
import argparse
import sys

sys.path.append("/data/mathieu/SummaFusion/src/") # todo: change to your folder path

from common.utils import *
from common.evaluation import *
from common.data import load_data
from dataset import *
from model import *
from engine import *
from model_utils import *



parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--cuda', type = bool, default = True)
parser.add_argument('--debug', type = bool, default = False)
parser.add_argument('--debug_size', type = int, default = 10)
parser.add_argument('--few_shot', type = bool, default = True)

# data
parser.add_argument('--dataset', type=str, default = "samsum",
                    choices= ["xsum", "reddit", "samsum"])

# model
parser.add_argument('--model_type', type = str, default = "pegasus",
                    choices=["pegasus"])
parser.add_argument('--model', type = str, default = "google/pegasus-large",
                    choices = ["google/pegasus-large", "google/pegasus-xsum"])
parser.add_argument('--model_name', type=str, default = "pegasus_samsum_train_100_seed_42_1",
                    choices = [
                        # XSum
                        ### full-shot
                        "pegasus_xsum_first_half_shuffled_1", "pegasus_xsum_second_half_shuffled_1", "pegasus_xsum",
                        ### 10-shot
                        "pegasus_xsum_first_half_train_10_seed_42_shuffled_1", "pegasus_xsum_second_half_train_10_seed_42_shuffled_1", "pegasus_xsum_train_10_seed_42_1",
                        "pegasus_xsum_first_half_train_10_seed_43_shuffled_1", "pegasus_xsum_second_half_train_10_seed_43_shuffled_1", "pegasus_xsum_train_10_seed_43_1",
                        "pegasus_xsum_first_half_train_10_seed_44_shuffled_1", "pegasus_xsum_second_half_train_10_seed_44_shuffled_1", "pegasus_xsum_train_10_seed_44_1",
                        ### 100-shot
                        "pegasus_xsum_first_half_train_100_seed_42_shuffled_1", "pegasus_xsum_second_half_train_100_seed_42_shuffled_1", "pegasus_xsum_train_100_seed_42_1",
                        "pegasus_xsum_first_half_train_100_seed_43_shuffled_1", "pegasus_xsum_second_half_train_100_seed_43_shuffled_1", "pegasus_xsum_train_100_seed_43_1",
                        "pegasus_xsum_first_half_train_100_seed_44_shuffled_1", "pegasus_xsum_second_half_train_100_seed_44_shuffled_1", "pegasus_xsum_train_100_seed_44_1",
                        ### 1000-shot
                        "pegasus_xsum_first_half_train_1000_seed_42_shuffled_1", "pegasus_xsum_second_half_train_1000_seed_42_shuffled_1", "pegasus_xsum_train_1000_seed_42_1",
                        "pegasus_xsum_first_half_train_1000_seed_43_shuffled_1", "pegasus_xsum_second_half_train_1000_seed_43_shuffled_1", "pegasus_xsum_train_1000_seed_43_1",
                        "pegasus_xsum_first_half_train_1000_seed_44_shuffled_1", "pegasus_xsum_second_half_train_1000_seed_44_shuffled_1", "pegasus_xsum_train_1000_seed_44_1",
                        # Reddit TIFU
                        ### full-shot
                        "pegasus_reddit_first_half_shuffled_1", "pegasus_reddit_second_half_shuffled_1", "pegasus_reddit_train_1",
                        ### 10-shot
                        "pegasus_reddit_first_half_train_10_seed_42_shuffled_1", "pegasus_reddit_second_half_train_10_seed_42_shuffled_1", "pegasus_reddit_train_10_seed_42_1",
                        "pegasus_reddit_first_half_train_10_seed_43_shuffled_1", "pegasus_reddit_second_half_train_10_seed_43_shuffled_1", "pegasus_reddit_train_10_seed_43_1",
                        "pegasus_reddit_first_half_train_10_seed_44_shuffled_1", "pegasus_reddit_second_half_train_10_seed_44_shuffled_1", "pegasus_reddit_train_10_seed_44_1",
                        ### 100-shot
                        "pegasus_reddit_first_half_train_100_seed_42_shuffled_1", "pegasus_reddit_second_half_train_100_seed_42_shuffled_1", "pegasus_reddit_train_100_seed_42_1",
                        "pegasus_reddit_first_half_train_100_seed_43_shuffled_1", "pegasus_reddit_second_half_train_100_seed_43_shuffled_1", "pegasus_reddit_train_100_seed_43_1",
                        "pegasus_reddit_first_half_train_100_seed_44_shuffled_1", "pegasus_reddit_second_half_train_100_seed_44_shuffled_1", "pegasus_reddit_train_100_seed_44_1",
                        ### 1000-shot
                        "pegasus_reddit_first_half_train_1000_seed_42_shuffled_1", "pegasus_reddit_second_half_train_1000_seed_42_shuffled_1", "pegasus_reddit_train_1000_seed_42_1",
                        "pegasus_reddit_first_half_train_1000_seed_43_shuffled_1", "pegasus_reddit_second_half_train_1000_seed_43_shuffled_1", "pegasus_reddit_train_1000_seed_43_1",
                        "pegasus_reddit_first_half_train_1000_seed_44_shuffled_1", "pegasus_reddit_second_half_train_1000_seed_44_shuffled_1", "pegasus_reddit_train_1000_seed_44_1",
                        # SAMSum
                        ### full-shot
                        "pegasus_samsum_first_half_shuffled_1", "pegasus_samsum_second_half_shuffled_1", "pegasus_samsum_train_1",
                        ### 10-shot
                        "pegasus_samsum_first_half_train_10_seed_42_shuffled_1", "pegasus_samsum_second_half_train_10_seed_42_shuffled_1", "pegasus_samsum_train_10_seed_42_1",
                        "pegasus_samsum_first_half_train_10_seed_43_shuffled_1", "pegasus_samsum_second_half_train_10_seed_43_shuffled_1", "pegasus_samsum_train_10_seed_43_1",
                        "pegasus_samsum_first_half_train_10_seed_44_shuffled_1", "pegasus_samsum_second_half_train_10_seed_44_shuffled_1", "pegasus_samsum_train_10_seed_44_1",
                        ### 100-shot
                        "pegasus_samsum_first_half_train_100_seed_42_shuffled_1", "pegasus_samsum_second_half_train_100_seed_42_shuffled_1", "pegasus_samsum_train_100_seed_42_1",
                        "pegasus_samsum_first_half_train_100_seed_43_shuffled_1", "pegasus_samsum_second_half_train_100_seed_43_shuffled_1", "pegasus_samsum_train_100_seed_43_1",
                        "pegasus_samsum_first_half_train_100_seed_44_shuffled_1", "pegasus_samsum_second_half_train_100_seed_44_shuffled_1", "pegasus_samsum_train_100_seed_44_1",
                        ### 1000-shot
                        "pegasus_samsum_first_half_train_1000_seed_42_shuffled_1", "pegasus_samsum_second_half_train_1000_seed_42_shuffled_1", "pegasus_samsum_train_1000_seed_42_1",
                        "pegasus_samsum_first_half_train_1000_seed_43_shuffled_1", "pegasus_samsum_second_half_train_1000_seed_43_shuffled_1", "pegasus_samsum_train_1000_seed_43_1",
                        "pegasus_samsum_first_half_train_1000_seed_44_shuffled_1", "pegasus_samsum_second_half_train_1000_seed_44_shuffled_1", "pegasus_samsum_train_1000_seed_44_1",
                    ])
parser.add_argument('--hidden_size', type = int, default = 768) # 768
parser.add_argument('--cache_dir', type = str,
                    default = "../../../hf_models/pegasus-large/")
parser.add_argument('--load_model', type = bool, default = True)
parser.add_argument('--load_model_path', type = str,
                    default = "pegasus_samsum_train_100_seed_42_1+90",
                    choices = [
                        # XSum
                        # full-shot
                        "pegasus_xsum_first_half_shuffled_2+1500", "pegasus_xsum_second_half_shuffled_2+1500",
                        ### 10-shot
                        "pegasus_xsum_first_half_train_10_seed_42_shuffled_1+15", "pegasus_xsum_second_half_train_10_seed_42_shuffled_1+15", "pegasus_xsum_train_10_seed_42_1+30",
                        "pegasus_xsum_first_half_train_10_seed_43_shuffled_1+15", "pegasus_xsum_second_half_train_10_seed_43_shuffled_1+15", "pegasus_xsum_train_10_seed_43_1+30",
                        "pegasus_xsum_first_half_train_10_seed_44_shuffled_1+15", "pegasus_xsum_second_half_train_10_seed_44_shuffled_1+15", "pegasus_xsum_train_10_seed_44_1+30",
                        ### 100-shot
                        "pegasus_xsum_first_half_train_100_seed_42_shuffled_1+40", "pegasus_xsum_second_half_train_100_seed_42_shuffled_1+40", "pegasus_xsum_train_100_seed_42_1+90",
                        "pegasus_xsum_first_half_train_100_seed_43_shuffled_1+40", "pegasus_xsum_second_half_train_100_seed_43_shuffled_1+40", "pegasus_xsum_train_100_seed_43_1+90",
                        "pegasus_xsum_first_half_train_100_seed_44_shuffled_1+40", "pegasus_xsum_second_half_train_100_seed_44_shuffled_1+40", "pegasus_xsum_train_100_seed_44_1+90",
                        ### 1000-shot
                        "pegasus_xsum_first_half_train_1000_seed_42_shuffled_1+100", "pegasus_xsum_second_half_train_1000_seed_42_shuffled_1+100", "pegasus_xsum_train_1000_seed_42_1+180",
                        "pegasus_xsum_first_half_train_1000_seed_43_shuffled_1+100", "pegasus_xsum_second_half_train_1000_seed_43_shuffled_1+100", "pegasus_xsum_train_1000_seed_43_1+180",
                        "pegasus_xsum_first_half_train_1000_seed_44_shuffled_1+100", "pegasus_xsum_second_half_train_1000_seed_44_shuffled_1+100", "pegasus_xsum_train_1000_seed_44_1+180",
                        # Reddit
                        # full-shot
                        "pegasus_reddit_first_half_shuffled_2+900", "pegasus_reddit_second_half_shuffled_2+700", "pegasus_reddit_train_4+1250",
                        ### 10-shot
                        "pegasus_reddit_first_half_train_10_seed_42_shuffled_1+14", "pegasus_reddit_second_half_train_10_seed_42_shuffled_1+14", "pegasus_reddit_train_10_seed_42_1+30",
                        "pegasus_reddit_first_half_train_10_seed_43_shuffled_1+14", "pegasus_reddit_second_half_train_10_seed_43_shuffled_1+14", "pegasus_reddit_train_10_seed_43_1+30",
                        "pegasus_reddit_first_half_train_10_seed_44_shuffled_1+14", "pegasus_reddit_second_half_train_10_seed_44_shuffled_1+14", "pegasus_reddit_train_10_seed_44_1+30",
                        ### 100-shot
                        "pegasus_reddit_first_half_train_100_seed_42_shuffled_1+40", "pegasus_reddit_second_half_train_100_seed_42_shuffled_1+40", "pegasus_reddit_train_100_seed_42_1+90",
                        "pegasus_reddit_first_half_train_100_seed_43_shuffled_1+40", "pegasus_reddit_second_half_train_100_seed_43_shuffled_1+40", "pegasus_reddit_train_100_seed_43_1+90",
                        "pegasus_reddit_first_half_train_100_seed_44_shuffled_1+40", "pegasus_reddit_second_half_train_100_seed_44_shuffled_1+40", "pegasus_reddit_train_100_seed_44_1+90",
                        ### 1000-shot
                        "pegasus_reddit_first_half_train_1000_seed_42_shuffled_1+100", "pegasus_reddit_second_half_train_1000_seed_42_shuffled_1+100", "pegasus_reddit_train_1000_seed_42_1+180",
                        "pegasus_reddit_first_half_train_1000_seed_43_shuffled_1+100", "pegasus_reddit_second_half_train_1000_seed_43_shuffled_1+100", "pegasus_reddit_train_1000_seed_43_1+180",
                        "pegasus_reddit_first_half_train_1000_seed_44_shuffled_1+100", "pegasus_reddit_second_half_train_1000_seed_44_shuffled_1+100", "pegasus_reddit_train_1000_seed_44_1+180",
                        # SAMSum
                        # full-shot
                        "pegasus_samsum_first_half_shuffled_2+550", "pegasus_samsum_second_half_shuffled_2+400", "pegasus_samsum_train_4+1200",
                        ### 10-shot
                        "pegasus_samsum_first_half_train_10_seed_42_shuffled_1+15", "pegasus_samsum_second_half_train_10_seed_42_shuffled_1+15", "pegasus_samsum_train_10_seed_42_1+30",
                        "pegasus_samsum_first_half_train_10_seed_43_shuffled_1+15", "pegasus_samsum_second_half_train_10_seed_43_shuffled_1+15", "pegasus_samsum_train_10_seed_43_1+30",
                        "pegasus_samsum_first_half_train_10_seed_44_shuffled_1+15", "pegasus_samsum_second_half_train_10_seed_44_shuffled_1+15", "pegasus_samsum_train_10_seed_44_1+30",
                        ### 100-shot
                        "pegasus_samsum_first_half_train_100_seed_42_shuffled_1+40", "pegasus_samsum_second_half_train_100_seed_42_shuffled_1+40", "pegasus_samsum_train_100_seed_42_1+90",
                        "pegasus_samsum_first_half_train_100_seed_43_shuffled_1+40", "pegasus_samsum_second_half_train_100_seed_43_shuffled_1+40", "pegasus_samsum_train_100_seed_43_1+90",
                        "pegasus_samsum_first_half_train_100_seed_44_shuffled_1+40", "pegasus_samsum_second_half_train_100_seed_44_shuffled_1+40", "pegasus_samsum_train_100_seed_44_190",
                        ### 1000-shot
                        "pegasus_samsum_first_half_train_1000_seed_42_shuffled_1+100", "pegasus_samsum_second_half_train_1000_seed_42_shuffled_1+100", "pegasus_samsum_train_1000_seed_42_1+190",
                        "pegasus_samsum_first_half_train_1000_seed_43_shuffled_1+100", "pegasus_samsum_second_half_train_1000_seed_43_shuffled_1+100", "pegasus_samsum_train_1000_seed_43_1+180",
                        "pegasus_samsum_first_half_train_1000_seed_44_shuffled_1+100", "pegasus_samsum_second_half_train_1000_seed_44_shuffled_1+100", "pegasus_samsum_train_1000_seed_44_1+180",
                    ]) # todo: change to where you saved the finetuned checkpoint

# summary generation
parser.add_argument('--val_dataset', type=str, default = "val_100_seed_42",
                    choices = [
                        # full-shot
                        "train", "first_half_train_shuffled", "second_half_train_shuffled", "val", "test",
                        # 10-shot
                        "first_half_train_10_seed_42_shuffled", "second_half_train_10_seed_42_shuffled", "val_10_seed_42",
                        "first_half_train_10_seed_43_shuffled", "second_half_train_10_seed_43_shuffled", "val_10_seed_43",
                        "first_half_train_10_seed_44_shuffled", "second_half_train_10_seed_44_shuffled", "val_10_seed_44",
                        # 100-shot
                        "first_half_train_100_seed_42_shuffled", "second_half_train_100_seed_42_shuffled", "val_100_seed_42",
                        "first_half_train_100_seed_43_shuffled", "second_half_train_100_seed_43_shuffled", "val_100_seed_43",
                        "first_half_train_100_seed_44_shuffled", "second_half_train_100_seed_44_shuffled", "val_100_seed_44",
                        # 1000-shot
                        "first_half_train_1000_seed_42_shuffled", "second_half_train_1000_seed_42_shuffled", "val_1000_seed_42",
                        "first_half_train_1000_seed_43_shuffled", "second_half_train_1000_seed_43_shuffled", "val_1000_seed_43",
                        "first_half_train_1000_seed_44_shuffled", "second_half_train_1000_seed_44_shuffled", "val_1000_seed_44",
                    ])
parser.add_argument('--max_val_size', type = int, default = 100000)
parser.add_argument('--inference_bs', type = int, default = 2) 
parser.add_argument('--save_summaries', type = bool, default = True)
parser.add_argument('--generation_method', type = str, default = "diverse_beam_search",
                    choices = ["beam_search", "diverse_beam_search", "top_p_sampling", "top_k_sampling"])
parser.add_argument('--num_return_sequences', type = int, default = 15) # default: 15
parser.add_argument('--num_beams', type = int, default = 15) # for beam search
parser.add_argument('--num_beam_groups', type = int, default = 15) # for diverse beam search
parser.add_argument('--diversity_penalty', type = float, default = 1.0) # for diverse beam search
parser.add_argument('--top_p', type = float, default = 0.95) # for top-p sampling
parser.add_argument('--top_k', type = int, default = 50) # for top-k sampling
parser.add_argument('--stemmer', type = bool, default = True)

# metrics 
parser.add_argument('--eval_rouge', type = bool, default = True)
parser.add_argument('--eval_bertscore', type = bool, default = False)
parser.add_argument('--eval_bartscore', type = bool, default = False)
parser.add_argument('--eval_new_ngram', type = bool, default = True)
parser.add_argument('--eval_rouge_text', type = bool, default = False)

args = parser.parse_args()

args.load_model_path = (args.load_model_path.split("+")[0], args.load_model_path.split("+")[1])

dataset_names = ["xsum", "reddit", "samsum"]
highlights = [False, False, False]
max_lengths = [512, 512, 512]
max_summary_lengths = [64, 64, 64]
clean_ns = [False, False, False]
length_penalties = [0.8, 0.6, 0.8]
repetition_penalties = [1.0, 1.0, 1.0]
no_repeat_ngram_sizes = [3, 3, 0]

idx = dataset_names.index(args.dataset)

args.highlights = highlights[idx]
args.max_length = max_lengths[idx]
args.max_summary_length = max_summary_lengths[idx]
args.clean_n = clean_ns[idx]
args.length_penalty = length_penalties[idx]
args.repetition_penalty = repetition_penalties[idx]
args.no_repeat_ngram_size = no_repeat_ngram_sizes[idx]

print("*"*50)
print(args)



def main(args):
    # seed
    seed_everything(args.seed)

    if not(os.path.isdir("../../summaries/")):
        os.makedirs("../../summaries/")
    if not(os.path.isdir("../../summaries/{}/".format(args.dataset))):
        os.makedirs("../../summaries/{}/".format(args.dataset))
    if not(os.path.isdir("../../summaries/{}/{}/".format(args.dataset, args.val_dataset))):
        os.makedirs("../../summaries/{}/{}/".format(args.dataset, args.val_dataset))
    if not(os.path.isdir("../../summaries/{}/{}/{}/".format(args.dataset, args.val_dataset, args.generation_method))):
        os.makedirs("../../summaries/{}/{}/{}/".format(args.dataset, args.val_dataset, args.generation_method))

    # device
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    args.device = device
    print("\nUsing device {}".format(device))

    # data
    val_data = load_data(args.val_dataset, args, individual_txt = args.highlights)

    # tokenizer
    tokenizer = build_tokenizer(args)

    # datasets
    mode = "val"
    texts, summaries = val_data
    print(len(texts), len(summaries))
    texts = texts[:args.max_val_size]
    summaries = summaries[:args.max_val_size]
    print(len(texts), len(summaries))
    if args.debug:
        texts = texts[:args.debug_size]
        summaries = summaries[:args.debug_size]
    val_dataset = Dataset(mode, tokenizer, texts, summaries, args)
    print("Total size of dataset: {}".format(len(texts)))

    # data loader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.inference_bs, shuffle = False)

    # model
    model = build_model(args)
    model = FTModel(model, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nThe model has {} trainable parameters".format(n_params))
    model = model.to(device)
    if args.load_model:
        (model_name, ckpt) = args.load_model_path
        ft = ""
        if args.few_shot:
            ft = "few_shot_"
        model_path = "../base_model_finetuning/{}ft_saved_models/{}/{}/checkpoint-{}/pytorch_model.bin".format(
            ft, args.dataset, model_name, ckpt
        )
        model.load_state_dict(torch.load(model_path))
        print("Loaded the model weights!", model_path)

    # summary generation
    val_texts, val_summaries, val_labels = get_summaries(tokenizer, val_loader, model, device, args)

    # evaluation
    base_results = [val_summaries[i][0] for i in range(len(val_summaries))]
    print("*"*100)
    print("\nTop beam:")
    overall_eval(val_texts, base_results, val_labels, args)

    # export
    num_candidates = len(val_summaries[0])
    if args.save_summaries:
        path = "../../summaries/{}/{}/{}/".format(args.dataset, args.val_dataset, args.generation_method)
        with open(path + "{}_texts_{}_beams_{}.pkl".format(args.val_dataset, len(val_texts), num_candidates), "wb") as f:
            pickle.dump(val_texts, f)
        with open(path + "{}_summaries_{}_{}_beams_{}.pkl".format(args.val_dataset, args.model_name, len(val_texts), num_candidates), "wb") as f:
            pickle.dump(val_summaries, f)
        with open(path + "{}_labels_{}_beams_{}.pkl".format(args.val_dataset, len(val_texts), num_candidates), "wb") as f:
            pickle.dump(val_labels, f)
        print("saved generated summaries!", path)


if __name__ == '__main__':

    main(args)
