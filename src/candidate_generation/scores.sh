# ROUGE-1/2/L scoring on the validation set

python main_scores.py \
--dataset samsum \
--val_dataset val_100_seed_42 \
--generation_method diverse_beam_search \
--model_name pegasus_samsum_train_100_seed_42_1 \
--num_candidates 15 \
--label_metric rouge_1 \
--save_scores True \

python main_scores.py \
--dataset samsum \
--val_dataset val_100_seed_42 \
--generation_method diverse_beam_search \
--model_name pegasus_samsum_train_100_seed_42_1 \
--num_candidates 15 \
--label_metric rouge_2 \
--save_scores True \

python main_scores.py \
--dataset samsum \
--val_dataset val_100_seed_42 \
--generation_method diverse_beam_search \
--model_name pegasus_samsum_train_100_seed_42_1 \
--num_candidates 15 \
--label_metric rouge_l \
--save_scores True \

