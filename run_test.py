import io
import os
import argparse
import csv

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, set_seed, AdamW, get_linear_schedule_with_warmup

from dataset_constructor import OneStopEnglishDataset, NewselaDataset, CambridgeEnglishReadabilityDataset, CommonCoreStandardsDataset
from tester import TesterForSeq2Seq
from utils.test_arguments import predefined_args


'''0. Arguments'''
parser = predefined_args(argparse.ArgumentParser())
args = parser.parse_args()



'''1. Reproducibility'''
set_seed(2022)



'''2. Look for GPU or use CPU'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



'''3. Model-specific configurations'''
print('Loading configuration...')
if not args.config:
    model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path)
else:
    model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.config)



'''4. Load model and tokenizer'''
print('Loading tokenizer and model...')
if not args.tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path, config=model_config)
else:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.tokenizer, config=model_config)

model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path, config=model_config)
model.to(device)



'''5. Make data pipeline'''
if args.dataset == "onestopenglish":
    test_dataset = OneStopEnglishDataset(
        tokenizer = tokenizer,
        split = 'test',
        max_seq_length = args.max_seq_length,
        )
elif args.dataset == "newsela":
    test_dataset = NewselaDataset(
        tokenizer = tokenizer,
        split = 'test',
        max_seq_length = args.max_seq_length,
        )
elif args.dataset == "cambridge":
    test_dataset = CambridgeEnglishReadabilityDataset(
        tokenizer = tokenizer,
        split = 'test',
        max_seq_length = args.max_seq_length,
        )
elif args.dataset == "commoncore":
    test_dataset = CommonCoreStandardsDataset(
        tokenizer = tokenizer,
        split = 'test',
        max_seq_length = args.max_seq_length,
        )

print(f'Created `test_dataset` of {args.dataset}, with {len(test_dataset)} examples!')

test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
print(f'Created `test_dataloader` with {len(test_dataloader)} batches!')


'''6. Test'''
tester = TesterForSeq2Seq(
    model = model,
    tokenizer = tokenizer,
    eval_dataloader = test_dataloader,
    device = device,    
    )
output = tester.test(
    return_true_labels = True, 
    return_pred_labels = False,
    )