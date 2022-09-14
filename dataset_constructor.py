import csv
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
import torch

from utils import file_handler as f_handler


"""
Parent Classes
    - GeneralInputExample
    - GeneralProcessor
    - GeneralDataset(Dataset)
"""
@dataclass(frozen=True)
class GeneralInputExample:
    text: str
    text_pair: Optional[str]
    label: Optional[str]

class GeneralProcessor:
    """Parent dataset processor"""

    def get_train_instances(self):
        """Process Train Set"""
        return NotImplementedError()

    def get_valid_instances(self):
        """Process Valid Set"""
        return NotImplementedError()

    def get_test_instances(self):
        """Process Test Set"""
        return NotImplementedError()

    def get_labels(self):
        """See base class."""
        return ["True", "False"]

    def _read_csv(self, input_file):
        df = f_handler.load_ndjson_and_return_pandas(input_file)
        return df.to_dict('records')

    def _create_instances(self, lines: List[dict]):
        """Creates instances for the training and dev sets."""
        instances = []
        for line_dict in lines:
            instances.append(
                GeneralInputExample(
                    text = line_dict['text'],
                    text_pair = line_dict['text_pair'],
                    label = line_dict['difficult_text']
                )
            )
        return instances


class DatasetForSeq2Seq(Dataset):
    def __init__(self, tokenizer, split, max_seq_length):
        self.split = split
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.proc = GeneralProcessor()
        self._build()
  
    def __getitem__(self, item):
        return {key: self.inputs[key][item] for key in self.inputs.keys()}
  
    def __len__(self):
        return len(self.texts)
  
    def _build(self):
        if self.split == 'train':
            self.instances = self.proc.get_train_instances()
        elif self.split == "valid":
            self.instances = self.proc.get_valid_instances()
        elif self.split == "test":
            self.instances = self.proc.get_test_instances()
        else:
            raise Exception("wrong split")

        self.texts, self.labels = self._iterate_and_prepare_instances()
        self._make_inputs()
    
    def _truncate_text(self, text: str) -> str:
        max_seq_length = int(self.max_seq_length/2 - 10)
        tokenized = self.tokenizer(
            text,
            max_length = max_seq_length,
            truncation=True,
            pad_to_max_length = False, 
            return_tensors = "pt"
            )
        text = self.tokenizer.decode(
            *tokenized['input_ids'].tolist(), 
            skip_special_tokens = True,
            )
        return text

    def _map_label(self, label) -> str:
        """label_mapper = {
            'text': 'Text 1', 
            'text_pair': 'Text 2'
        }# train"""
        label_mapper = {
            'text': 'Text 2', 
            'text_pair': 'Text 1'
        }# test"""
        return label_mapper[label]

    def _iterate_and_prepare_instances(self) -> Tuple[List[str], List[str]]:
        texts = []
        labels = []
        for instance in self.instances:
            text = self._truncate_text(instance.text)
            text_pair = self._truncate_text(instance.text_pair)
            label = self._map_label(instance.label)

            """texts.append(f"Which Text is more difficult? Text 1: {text} Text 2: {text_pair}") # train"""
            texts.append(f"Text 1: {text} Text 2: {text_pair} Easier:") # test"""
            labels.append(label)
        return texts, labels

    def _make_inputs(self):
        self.inputs = self.tokenizer(
            self.texts, 
            max_length=self.max_seq_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
            )
        self.labels = self.tokenizer(
            self.labels, 
            max_length = 5, 
            padding='max_length',
            return_tensors = "pt"
            )
        self.inputs.update({
            'labels':self.labels['input_ids']
        })



"""
OneStopEnglish
    - OneStopEnglishProcessor(GeneralProcessor)
    - OneStopEnglishDataset(DatasetForSeq2Seq)
"""
class OneStopEnglishProcessor(GeneralProcessor):
    """Processor for the OneStopEnglish."""

    def get_train_instances(self):
        """Process Train Set"""
        return self._create_instances(
            self._read_csv(
                "datasets/final_OSEN_train.json"
                )
        )

    def get_valid_instances(self):
        """Process Valid Set"""
        return self._create_instances(
            self._read_csv(
                "datasets/final_OSEN_dev.json"
                )
        )

    def get_test_instances(self):
        """Process Test Set"""
        return self._create_instances(
            self._read_csv(
                "datasets/final_OSEN_test.json"
                )
        )


class OneStopEnglishDataset(DatasetForSeq2Seq):
    """Dataset for the OneStopEnglish."""
    def __init__(self, tokenizer, split, max_seq_length):
        self.split = split
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.proc = OneStopEnglishProcessor()
        self._build()



"""
Newsela
    - NewselaProcessor(GeneralProcessor)
    - NewselaDataset(DatasetForSeq2Seq)
"""
class NewselaProcessor(GeneralProcessor):
    """Processor for the Newsela."""

    def get_train_instances(self):
        """Process Train Set"""
        return self._create_instances(
            self._read_csv(
                "datasets/final_NEWS_train.json"
                )
        )

    def get_valid_instances(self):
        """Process Valid Set"""
        return self._create_instances(
            self._read_csv(
                "datasets/final_NEWS_dev.json"
                )
        )

    def get_test_instances(self):
        """Process Test Set"""
        return self._create_instances(
            self._read_csv(
                "datasets/final_NEWS_test.json"
                )
        )


class NewselaDataset(DatasetForSeq2Seq):
    """Dataset for the Newsela."""
    def __init__(self, tokenizer, split, max_seq_length):
        self.split = split
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.proc = NewselaProcessor()
        self._build()



"""
CambridgeEnglishReadability
    - CambridgeEnglishReadabilityProcessor(GeneralProcessor)
    - CambridgeEnglishReadabilityDataset(DatasetForSeq2Seq)
"""
class CambridgeEnglishReadabilityProcessor(GeneralProcessor):
    """Processor for the CambridgeEnglishReadability."""

    def get_train_instances(self):
        """Process Train Set"""
        return self._create_instances(
            self._read_csv(
                "datasets/final_CAMB.json"
                )
        )

    def get_valid_instances(self):
        """Process Valid Set"""
        return self._create_instances(
            self._read_csv(
                "datasets/final_CAMB.json"
                )
        )

    def get_test_instances(self):
        """Process Test Set"""
        return self._create_instances(
            self._read_csv(
                "datasets/final_CAMB.json"
                )
        )


class CambridgeEnglishReadabilityDataset(DatasetForSeq2Seq):
    """Dataset for the CambridgeEnglishReadability."""
    def __init__(self, tokenizer, split, max_seq_length):
        self.split = split
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.proc = CambridgeEnglishReadabilityProcessor()
        self._build()



"""
CommonCoreStandards
    - CommonCoreStandardsProcessor(GeneralProcessor)
    - CommonCoreStandardsDataset(DatasetForSeq2Seq)
"""
class CommonCoreStandardsProcessor(GeneralProcessor):
    """Processor for the CommonCoreStandards."""

    def get_train_instances(self):
        """Process Train Set"""
        return self._create_instances(
            self._read_csv(
                "datasets/final_CCSB_0_3.json"
                )
        )

    def get_valid_instances(self):
        """Process Valid Set"""
        return self._create_instances(
            self._read_csv(
                "datasets/final_CCSB_0_3.json"
                )
        )

    def get_test_instances(self):
        """Process Test Set"""
        return self._create_instances(
            self._read_csv(
                "datasets/final_CCSB_0_3.json"
                )
        )


class CommonCoreStandardsDataset(DatasetForSeq2Seq):
    """Dataset for the CommonCoreStandards."""
    def __init__(self, tokenizer, split, max_seq_length):
        self.split = split
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.proc = CommonCoreStandardsProcessor()
        self._build()



if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="t5-base")

    test = OneStopEnglishDataset(
            tokenizer = tokenizer,
            split = 'train',
            max_seq_length = 512,
            )
    print(len(test))
    print(test[0])
    test = NewselaDataset(
            tokenizer = tokenizer,
            split = 'train',
            max_seq_length = 512,
            )
    print(len(test))
    print(test[0])
    test = CambridgeEnglishReadabilityDataset(
            tokenizer = tokenizer,
            split = 'train',
            max_seq_length = 512,
            )
    print(len(test))
    print(test[0])
    test = CommonCoreStandardsDataset(
            tokenizer = tokenizer,
            split = 'train',
            max_seq_length = 512,
            )
    print(len(test))
    print(test[0])