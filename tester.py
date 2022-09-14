from typing import Optional, Union, Tuple

from transformers import AutoModelForSequenceClassification
import torch

from utils.logger import ClassificationLogger
from utils.early_stopper import EarlyStopper

class TesterForSeq2Seq:
    def __init__(
        self,
        model: object,
        tokenizer: object,
        eval_dataloader: Optional[object]=None,
        device: Optional[torch.device]=None,
        ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataloader = eval_dataloader
        self.device = device
    
    def test(
        self,
        return_true_labels: bool = False, 
        return_pred_labels: bool = False,
        ) -> Tuple[Optional[float], Optional[list], Optional[list]]:

        """1. Start evaluation"""
        TestLogger = ClassificationLogger(
            name = f"ccsb-0-2-RF-test-bart-base-osen-1e-5-8-30", 
            len_batch = len(self.eval_dataloader), 
            num_epochs = 1, 
            interval = 50
            )
        self.model.eval()
        for idx, batch in enumerate(self.eval_dataloader):
            """2. Move batch to device"""
            batch = {k: v.to(self.device) for k, v in batch.items()}

            """3. Generate prediction labels"""
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids = batch['input_ids'],
                    attention_mask = batch['attention_mask'],
                    max_new_tokens = 10
                    )   

            """4. Decode prediction and true labels"""
            self.true_labels = self.tokenizer.batch_decode(
                batch['labels'].tolist(), 
                skip_special_tokens = True,
                )
            self.pred_labels = self.tokenizer.batch_decode(
                output_ids.tolist(), 
                skip_special_tokens = True,
                )
            TestLogger.record_end_batch(
                idx = idx, 
                pred_labels = self.pred_labels, 
                true_labels = self.true_labels,
                )
            
        f1, acc, prec, recl = TestLogger.record_end_epoch(return_metric = True)

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