# fine-tune model on the 1st half of the training set
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2000 main_trainer.py \
--dataset samsum \
--train_dataset first_half_train_100_seed_42_shuffled \
--max_train_size 50 \
--val_dataset val_100_seed_42 \
--max_val_size 100 \
--save_model_path few_shot_ft_saved_models/samsum/pegasus_samsum_first_half_train_100_seed_42_shuffled_1 \

# fine-tune model on the 2nd half of the training set
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2000 main_trainer.py \
--dataset samsum \
--train_dataset second_half_train_100_seed_42_shuffled \
--max_train_size 50 \
--val_dataset val_100_seed_42 \
--max_val_size 100 \
--save_model_path few_shot_ft_saved_models/samsum/pegasus_samsum_second_half_train_100_seed_42_shuffled_1 \

# fine-tune model on the entire training set (for Reddit)
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2000 main_trainer.py \
--dataset samsum \
--train_dataset train_100_seed_42 \
--max_train_size 100 \
--val_dataset val_100_seed_42 \
--max_val_size 100 \
--save_model_path few_shot_ft_saved_models/samsum/pegasus_samsum_train_100_seed_42_1 \
