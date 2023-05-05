# Prompt-based Learning for Text Readability Assessment

## Overview

This repository hosts research code from our research paper "Prompt-based Learning for Text Readability Assessment" at (EACL 2023)[https://aclanthology.org/2023.findings-eacl.135/]. You can train and evaluate models using the code here. The included scripts are self-explanatory with comments for easy reading!

## Installation

Download Repo

```bash
git clone https://github.com/brucewlee/lingfeat
```

Install Dependencies
```bash
pip install -r requirements.txt
```

Download Dataset
unzip and place all in datasets/ directory

## Training

Set arguments in utils/train_arguments.py

## Testing

Set arguments in utils/test_arguments.py


## Citation

```
@inproceedings{lee-lee-2023-prompt,
    title = "Prompt-based Learning for Text Readability Assessment",
    author = "Lee, Bruce W.  and
      Lee, Jason",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2023",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-eacl.135",
    pages = "1819--1824",
    abstract = "We propose the novel adaptation of a pre-trained seq2seq model for readability assessment. We prove that a seq2seq model - T5 or BART - can be adapted to discern which text is more difficult from two given texts (pairwise). As an exploratory study to prompt-learn a neural network for text readability in a text-to-text manner, we report useful tips for future work in seq2seq training and ranking-based approach to readability assessment. Specifically, we test nine input-output formats/prefixes and show that they can significantly influence the final model performance.Also, we argue that the combination of text-to-text training and pairwise ranking setup 1) enables leveraging multiple parallel text simplification data for teaching readability and 2) trains a neural model for the general concept of readability (therefore, better cross-domain generalization). At last, we report a 99.6{\%} pairwise classification accuracy on Newsela and a 98.7{\%} for OneStopEnglish, through a joint training approach. Our code is available at github.com/brucewlee/prompt-learning-readability.",
}
```
*Please cite our paper and provide link to this repository* if you use in this software in research.
