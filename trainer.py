from typing import Optional, Union, Tuple

from transformers import AutoModelForSequenceClassification
import torch

from utils.logger import ClassificationLogger
from utils.early_stopper import EarlyStopper

class TrainerForSeq2Seq:
    def __init__(
        self,
        model: object,
        tokenizer: object,
        train_dataloader: Optional[object]=None,
        eval_dataloader: Optional[object]=None,
        optimizer: Optional[object]=None,
        scheduler: Optional[object]=None,
        num_epochs: Optional[int]=None,
        device: Optional[torch.device]=None,
        early_stop: bool = False,
        ) -> None:

        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.early_stop = early_stop

        self.EarlyStopper = EarlyStopper("increase") 
        self.TrainLogger = ClassificationLogger(
            "news-Q-train-t5-base-1e-5-8-3", 
            len(train_dataloader), 
            self.num_epochs, 
            50
            )

    def train(
        self, 
        return_model: bool = False, 
        return_true_labels: bool = False, 
        return_pred_labels: bool = False
        ) -> Tuple[Optional[object], Optional[float], Optional[list], Optional[list]]:

        for epoch in range(self.num_epochs):
            self.model.train()
            for idx, batch in enumerate(self.train_dataloader):
                # self.true_labels = batch["labels"].numpy().flatten().tolist()

                batch = {k: v.type(torch.long).to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                self.model.zero_grad()
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                
                self.loss = loss.item()
                # self.pred_labels = logits.argmax(axis = -1).flatten().tolist()

                #self.TrainLogger.record_end_batch(idx, loss, self.pred_labels, self.true_labels)
                self.TrainLogger.record_end_batch(idx, self.loss)
            self.TrainLogger.record_end_epoch()
            self.validate()

            if self.continue_train == False:
                break

        to_return = self.construct_return(return_model, return_true_labels, return_pred_labels)
        return to_return
    
    def validate(
        self,
        return_true_labels: bool = False, 
        return_pred_labels: bool = False,
        mode: str = "valid"
        ) -> Tuple[Optional[float], Optional[list], Optional[list]]:

        """1. Start evaluation"""
        ValidLogger = ClassificationLogger(
            name = f"news-Q-{mode}-t5-base-1e-5-8-3", 
            len_batch = len(self.eval_dataloader), 
            num_epochs = 1, 
            interval = 50
            )
        self.model.eval()
        for idx, batch in enumerate(self.eval_dataloader):
            """2. Move batch to device"""
            #self.true_labels = batch["labels"].numpy().flatten().tolist()
            batch = {k: v.to(self.device) for k, v in batch.items()}

            """3. Generate prediction labels"""
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids = batch['input_ids'],
                    attention_mask = batch['attention_mask'],
                    max_new_tokens = 10
                    )   
                """outputs = self.model(**batch)
                self.loss = outputs.loss.item()
                logits = outputs.logits"""

            """4. Decode prediction and true labels"""
            #self.pred_labels = logits.argmax(axis = -1).flatten().tolist()
            self.true_labels = self.tokenizer.batch_decode(
                batch['labels'].tolist(), 
                skip_special_tokens = True,
                )
            self.pred_labels = self.tokenizer.batch_decode(
                output_ids.tolist(), 
                skip_special_tokens = True,
                )
            ValidLogger.record_end_batch(
                idx = idx, 
                pred_labels = self.pred_labels, 
                true_labels = self.true_labels,
                )
            #print(self.true_labels)
            #print(self.pred_labels)
            
        f1, acc, prec, recl = ValidLogger.record_end_epoch(return_metric = True)
        if self.early_stop:
            self.continue_train = self.EarlyStopper.check(to_track = f1)
        else:
            self.continue_train = True

        to_return = self.construct_return(return_true_labels, return_pred_labels)
        return to_return

    def save(self, name) -> None:
        self.model.save_pretrained(f"ckpt/{name}-{self.num_epochs}epochs")
    
    def construct_return(
        self,
        return_model: bool = False, 
        return_true_labels: bool = False, 
        return_pred_labels: bool = False
        ) -> list:

        to_return = {}
        if return_model:
            to_return['model'] = self.model
        if return_true_labels:
            to_return['true_labels'] = self.true_labels
        if return_pred_labels:
            to_return['pred_labels'] = self.pred_labels
        
        return to_return