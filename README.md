# SummaFusion
Source code for the paper <a href="https://arxiv.org/pdf/2210.08779.pdf" style = "text-decoration:none;color:#4682B4">Towards Summary Candidates Fusion</a>.

Mathieu Ravaut, Shafiq Joty, Nancy F. Chen.

Accepted for publication at EMNLP 2022. 

## Setup

### 1 - Download the code
```
git clone https://github.com/Ravoxsg/SummaFusion.git
cd SummaFusion
```

### 2 - Install the dependencies
```
conda create --name summa_fusion python=3.8.8
conda activate summa_fusion
pip install -r requirements.txt
```

## Dataset

We use HuggingFace datasets library to access and save each dataset.
We save it as .txt file for the sources, and another one for the summaries, with 1 data point per line.

For instance to download and save SAMSum (default code):
```
cd src/candidate_generation/
bash dataset.sh
```

Note that for Reddit TIFU, we make a custom 80/10/10 train/val/test split.  
To match our results on Reddit TIFU, first double check that you have the following:  
For **training set**, size is **33,704** and the first data point summary is:  
*got a toy train from first grade. used an old hot wheels ramp to fling it into the air and smash my ceiling fan globe.*  
For the **validation** set, size is **4,213** and the first data point summary is:  
*married a redditor.  created a reddit account.  lost many hours to reddit.*  
For the **test** set, size is **4,222** and the first data point summary is:  
*laughed at baby boner...it turned into a super soaker.*  

If you want to work in few-shot, you need to prepare the (train, val) few-shot pairs. For each dataset and each few-shot size (among {10,100,1000}), we sample 3 pairs, corresponding to seeds {42,43,44}.

For instance on SAMSum 100-shot (default code):
```
bash few_shot.sh
```

## DEMO 
If you just want a demo (in a single file) of SummaFusion on a single data point (default: XSum), run:
```
cd src/summafusion/
CUDA_VISIBLE_DEVICES=0 python demo.py
```

## EVALUATION pipeline 

### 1 - Generate summary candidates
SummaFusion takes as input a set of summary candidates from a given sequence-to-sequence model PEGASUS with diverse beam search.

You need such a fine-tuned checkpoint before generating the candidates. 

For instance on SAMSum 100-shot validation set (default code):
```
CUDA_VISIBLE_DEVICES=0 bash candidate_generation.sh
```
Generating summary candidates should take a few minutes in few-shot, and up to a few hours on the full validation or test sets of XSum, Reddit or SAMSum.

### 2 - Score the candidates
As part of SummaFusion, we train a classifier on the summary candidates and thus need candidate-level information.

For instance to score candidates on SAMSum 100-shot validation set with ROUGE-1/2/L (default code):
```
bash scores.sh
```
Scoring all candidates should take a few seconds in few-shot, and up to a few minutes on the validation or test sets of XSum, Reddit or SAMSum. 

### 3 - Download the SummaFusion model checkpoint
XSum full-shot checkpoint: <a href="https://drive.google.com/file/d/1_6-Yj8vj7WNnXLFypEefIk1G0J4wDaQh/view?usp=share_link" style = "text-decoration:none;color:#4682B4">here</a>   
XSum 100-shot checkpoint (seed: 42): <a href="https://drive.google.com/file/d/14km59vaoH-qIGJNNoQ5QhnY4FK2nv9oP/view?usp=share_link" style = "text-decoration:none;color:#4682B4">here</a>   
Reddit full-shot checkpoint: <a href="https://drive.google.com/file/d/1QnSFLYDtm449irp4HjFyX_LvPsKOt4TF/view?usp=share_link" style = "text-decoration:none;color:#4682B4">here</a>  
Reddit 100-shot checkpoint (seed: 42): <a href="https://drive.google.com/file/d/1m-DiouvQGhkAAfu52Bx9-YWsq3l1hIBw/view?usp=share_link" style = "text-decoration:none;color:#4682B4">here</a>   
SAMSum full-shot checkpoint: <a href="https://drive.google.com/file/d/1_qZJGxduCKUB6C1egFgf5Coyo6s2OMOe/view?usp=share_link" style = "text-decoration:none;color:#4682B4">here</a>  
SAMSum 100-shot checkpoint (seed: 42): <a href="https://drive.google.com/file/d/1YwIwwtwVD-gH101_CgWBDd8lmjR7v0zH/view?usp=share_link" style = "text-decoration:none;color:#4682B4">here</a>   

If you are using a full-shot checkpoint, place it into:
```
src/summafusion/saved_models/{dataset}/
```
And if it is a few-shot checkpoint, place it into:
```
src/summafusion/saved_models/{dataset}_few_shot/
```
where {dataset} is in {xsum,reddit,samsum} corresponds to the dataset name. 

### 4 - Run SummaFusion
For instance, to run SummaFusion on SAMSum 100-shot validation set (default code):
```
cd ../summafusion/
CUDA_VISIBLE_DEVICES=0 bash evaluate.sh
```
Make sure that the argument --load_model_path points to the name of the checkpoint you want to use. 

## TRAINING pipeline

### 1 - Fine-tune base models

We follow a cross-validation approach similar to SummaReranker.

First we split each training set into two halves.

For instance on SAMSum 100-shot:
```
cd ../base_model_finetuning/
bash build_train_splits.sh
```

Then we train a model on each half, and a third model on the entire training set. 
```
CUDA_VISIBLE_DEVICES=0 bash train_base_models.sh
```

### 2 - Generate summary candidates
Then, we need to get summary candidates on the training, validation and test sets. 

For instance on SAMSum 100-shot:
```
cd ../candidate_generation/
CUDA_VISIBLE_DEVICES=0 bash candidate_generation_train.sh
```
Generating summary candidates should take a few minutes in few-shot, and up to a few days for XSum full-shot. 

### 3 - Score the candidates
Next, we need to score the summary candidates on the training, validation and test sets for each of the metrics. This is needed for the candidate-level classification part of SummaFusion. 

For instance on SAMSum 100-shot with ROUGE-1/2/L:
```
CUDA_VISIBLE_DEVICES=0 bash scores_train.sh
```
Scoring all candidates should take a few seconds in few-shot, a few minutes in full-shot. 

### 4 - Train SummaFusion
For instance, to train Summafusion on SAMSum 100-shot:
```
cd ../summafusion/
CUDA_VISIBLE_DEVICES=0 bash train.sh
```

## Citation
If you find our paper or this project helps your research, please kindly consider citing our paper in your publication.   
```
@article{ravaut2022towards,
  title={Towards Summary Candidates Fusion},
  author={Ravaut, Mathieu and Joty, Shafiq and Chen, Nancy F},
  journal={arXiv preprint arXiv:2210.08779},
  year={2022}
}

```
