import io
import os
import argparse
import csv

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, set_seed, AdamW, get_linear_schedule_with_warmup

from dataset_constructor import OneStopEnglishDataset
from trainer import TrainerForSeq2Seq
from utils.train_arguments import predefined_args


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
    train_dataset = OneStopEnglishDataset(
        tokenizer = tokenizer,
        split = 'train',
        max_seq_length = args.max_seq_length,
        )
    print(f'Created `train_dataset` of {args.dataset}, with {len(train_dataset)} examples!')
    valid_dataset = OneStopEnglishDataset(
        tokenizer = tokenizer,
        split = 'valid',
        max_seq_length = args.max_seq_length,
        )
    print(f'Created `valid_dataset` of {args.dataset}, with {len(valid_dataset)} examples!')

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
print(f'Created `train_dataloader` with {len(train_dataloader)} batches!')

valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
print(f'Created `valid_dataloader` with {len(valid_dataloader)} batches!')



'''6. Make optimizer and schedhuler'''
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
optimizer = AdamW(
    model.parameters(), 
    lr = args.learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
    )
total_steps = len(train_dataloader) * args.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)


'''7. Train'''
trainer = TrainerForSeq2Seq(
    model = model,
    tokenizer = tokenizer,
    train_dataloader = train_dataloader,
    eval_dataloader = valid_dataloader,
    optimizer = optimizer,
    scheduler = scheduler,
    num_epochs = args.epochs,
    device = device,    
    )
trainer.train()
if args.save:
    name = args.model_name_or_path.replace('/','-')
    trainer.save(f'{name}-{args.dataset}-{args.learning_rate}')