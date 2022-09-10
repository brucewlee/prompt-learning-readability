import time
import os
from typing import Tuple, Optional
import csv

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tabulate import tabulate

class ClassificationLogger:
    def __init__ (
        self, 
        name: str, 
        len_batch: int, 
        num_epochs: int, 
        interval: int, 
        record_tsv: bool = True
        ) -> None:

        self.start_time = time.time()
        self.prev_end_batch_time = self.start_time
        self.prev_end_epoch_time = self.start_time
        
        self.name = name
        self.current_epoch = 1
        self.num_epochs = num_epochs
        self.len_batch = len_batch
        self.interval = interval
        self.record_tsv = record_tsv

        # variables to accumulate
        self.epoch_loss = 0
        self.epoch_pred_labels = []
        self.epoch_true_labels = []

    def record_end_batch(
        self, 
        idx: int = 0, 
        loss: float = 0,
        pred_labels: list = [], 
        true_labels: list = [], 
        return_metric: bool = False
        ) -> Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]:

        self.idx = f'Batch {idx}'
        self.total = self.len_batch
        self.time_taken = time.time() - self.prev_end_batch_time

        self.epoch_loss += loss
        self.epoch_pred_labels.extend(pred_labels)
        self.epoch_true_labels.extend(true_labels)

        self.loss = loss
        self.pred_labels = pred_labels
        self.true_labels = true_labels

        if idx % self.interval == 0:
            self.calculate_metrics()
            self.make_report()
            if self.record_tsv:
                self.write_tsv()

        self.prev_end_batch_time = time.time()

        if return_metric:
            return self.f1, self.acc, self.prec, self.recl

    def record_end_epoch(
        self,
        return_metric: bool = False
        ) -> Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]:

        self.idx = f'Epoch {self.current_epoch}'
        self.total = self.num_epochs
        self.time_taken = time.time() - self.prev_end_epoch_time

        self.loss = self.epoch_loss / self.len_batch
        self.pred_labels = self.epoch_pred_labels
        self.true_labels = self.epoch_true_labels
        
        self.calculate_metrics()
        self.make_report()
        self.write_tsv()

        self.current_epoch += 1
        self.prev_end_epoch_time = time.time()

        if return_metric:
            return self.f1, self.acc, self.prec, self.recl

    def calculate_metrics(self):
        self.f1 = f1_score(self.true_labels, self.pred_labels, average='micro')
        self.acc = accuracy_score(self.true_labels, self.pred_labels)
        self.prec = precision_score(self.true_labels, self.pred_labels, average='micro')
        self.recl = recall_score(self.true_labels, self.pred_labels, average='micro')
    
    def make_report(self):
        headers = ['idx / total', 'Loss', 'Accuracy', 'F1', 'Time(s)', 'Total Time(s)']
        table = [
            [
                f"{self.name} -> {self.idx} / {self.total}", 
                f"{self.loss:.5f}",
                f"{self.acc:.5f}",
                f"{self.f1:.5f}",
                f"{self.time_taken:.5f}",
                f"{time.time() - self.start_time:.5f}"
            ],
            [

            ]
        ]
        print(tabulate(table, headers, tablefmt="outline"))
    
    def write_tsv(self):
        # loss, acc result save
        if os.path.isfile(f'{self.name}_result.tsv'):
            with open(f'{self.name}_result.tsv', 'a', newline='') as f:
                a = csv.writer(f, delimiter='\t')
                a.writerow([
                    self.loss, 
                    self.acc,
                    self.f1,
                    self.prec,
                    self.recl,
                    self.time_taken,
                    time.time() - self.start_time
                    ])
        else:
            with open(f'{self.name}_result.tsv', 'w', newline='') as f:
                w = csv.writer(f, delimiter='\t')
                w.writerow([
                    'loss',
                    'acc',
                    'f1',
                    'prec',
                    'recl',
                    'time(s)',
                    'total time(s)'
                    ])
                w.writerow([
                    self.loss, 
                    self.acc,
                    self.f1,
                    self.prec,
                    self.recl,
                    self.time_taken,
                    time.time() - self.start_time
                    ])