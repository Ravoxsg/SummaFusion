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

For instance to download and save SAMSum (default in the files):
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

If you want to work in few-shot, you need to prepare the (train, val) few-shot pairs.

For instance on SAMSum 100-shot:
```
bash few_shot.sh
```

## EVALUATION pipeline (assumes an already trained SummaFusion checkpoint)

### 1 - Generate summary candidates
SummaFusion takes as input a set of summary candidates from a given sequence-to-sequence model PEGASUS with diverse beam search.

You need such a fine-tuned checkpoint before generating the candidates. 

For instance on SAMSum 100-shot validation set:
```
CUDA_VISIBLE_DEVICES=0 bash candidate_generation.sh
```
Generating summary candidates should take a few minutes in few-shot, and up to a few hours on the full validation or test sets of XSum, Reddit or SAMSum.

### 2 - Score the candidates
As part of SummaFusion, we train a classifier on the summary candidates and thus need candidate-level information.

For instance to score candidates on SAMSum 100-shot validation set with ROUGE-1/2/L:
```
bash scores.sh
```
Scoring all candidates should take a few seconds in few-shot, and up to a few minutes on the validation or test sets of XSum, Reddit or SAMSum. 

### 3 - Download the model checkpoint
XSum full-shot checkpoint: <a href="link" style = "text-decoration:none;color:#4682B4">here</a>   
XSum 100-shot checkpoint: <a href="link" style = "text-decoration:none;color:#4682B4">here</a>   
Reddit full-shot checkpoint: <a href="link" style = "text-decoration:none;color:#4682B4">here</a>  
Reddit 100-shot checkpoint: <a href="link" style = "text-decoration:none;color:#4682B4">here</a>   
SAMSum full-shot checkpoint: <a href="link" style = "text-decoration:none;color:#4682B4">here</a>  
SAMSum 100-shot checkpoint: <a href="link" style = "text-decoration:none;color:#4682B4">here</a>   

### 4 - Run SummaFusion
For instance, to run SummaFusion on SAMSum 100-shot validation set:
```
cd ../summareranker/
CUDA_VISIBLE_DEVICES=0 bash evaluate.sh
```
Make sure that the argument --load_model_path points to where you placed the SummaFusion checkpoint. 
