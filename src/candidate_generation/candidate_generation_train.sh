# Model trained on 1st training half => infer on 2nd training half
python main_candidate_generation.py \
--dataset samsum \
--val_dataset second_half_train_100_seed_42_shuffled \
--model google/pegasus-large \
--model_name pegasus_samsum_first_half_train_100_seed_42_shuffled_1 \
--cache_dir ../../../hf_models/pegasus-large \
--load_model True \
--load_model_path pegasus_samsum_first_half_train_100_seed_42_shuffled_1+40 \
--inference_bs 4 \
--save_summaries True \
--generation_method diverse_beam_search \
--num_return_sequences 15 \
--num_beams 15 \
--num_beam_groups 15 \

# Model trained on 2nd training half => infer on 1st training half
python main_candidate_generation.py \
--dataset samsum \
--val_dataset first_half_train_100_seed_42_shuffled \
--model google/pegasus-large \
--model_name pegasus_samsum_second_half_train_100_seed_42_shuffled_1 \
--cache_dir ../../../hf_models/pegasus-large \
--load_model True \
--load_model_path pegasus_samsum_second_half_train_100_seed_42_shuffled_1+40 \
--inference_bs 4 \
--save_summaries True \
--generation_method diverse_beam_search \
--num_return_sequences 15 \
--num_beams 15 \
--num_beam_groups 15 \

# Model trained on entire training set => infer on validation set
python main_candidate_generation.py \
--dataset samsum \
--val_dataset val_100_seed_42 \
--model google/pegasus-large \
--model_name pegasus_samsum_train_100_seed_42_1 \
--cache_dir ../../../hf_models/pegasus-large \
--load_model True \
--load_model_path pegasus_samsum_train_100_seed_42_1+90 \
--inference_bs 4 \
--save_summaries True \
--generation_method diverse_beam_search \
--num_return_sequences 15 \
--num_beams 15 \
--num_beam_groups 15 \

# Model trained on entire training set => infer on test set
python main_candidate_generation.py \
--dataset samsum \
--val_dataset test \
--model google/pegasus-large \
--model_name pegasus_samsum_train_100_seed_42_1 \
--cache_dir ../../../hf_models/pegasus-large \
--load_model True \
--load_model_path pegasus_samsum_train_100_seed_42_1+90 \
--inference_bs 4 \
--save_summaries True \
--generation_method diverse_beam_search \
--num_return_sequences 15 \
--num_beams 15 \
--num_beam_groups 15 \
