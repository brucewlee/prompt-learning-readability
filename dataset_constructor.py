import csv
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
import torch


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
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_instances(self, lines: List[List[str]]):
        """Creates instances for the training and dev sets."""
        instances = [
            GeneralInputExample(
                text = line[1],
                text_pair = line[2],
                label = line[3]
            )
            for line in lines[1:]  # we skip the line with the column names
        ]
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
        label_mapper = {
            'Text 1': 'Text 1', 
            'Text 2': 'Text 2'
        }
        return label_mapper[label]

    def _iterate_and_prepare_instances(self) -> Tuple[List[str], List[str]]:
        texts = []
        labels = []
        for instance in self.instances:
            text = self._truncate_text(instance.text)
            text_pair = self._truncate_text(instance.text_pair)
            label = self._map_label(instance.label)

            texts.append(f"Which Text is more difficult? Text 1: {text} Text 2: {text_pair}")
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
                "datasets/OneStopEnglish/preprocessed_train.csv"
                )
        )

    def get_valid_instances(self):
        """Process Valid Set"""
        return self._create_instances(
            self._read_csv(
                "datasets/OneStopEnglish/preprocessed_valid.csv"
                )
        )

    def get_test_instances(self):
        """Process Test Set"""
        return self._create_instances(
            self._read_csv(
                "datasets/OneStopEnglish/preprocessed_test.csv"
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



if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="t5-base")

    test = OneStopEnglishDataset(
            tokenizer = tokenizer,
            split = 'train',
            max_seq_length = 512,
            )
    print(len(test))
    test._truncate_text(text="In homes and cafes across the country, a cup of tea, baked beans on toast and fish and chips have long played a key role in the British dining experience. But, the extent of a change in tastes over the generations has been captured in a dataset published recently in the National Food Survey, which was set up in 1940 by the government after concerns about health and access to food.  Despite the apparent British love of tea, consumption has more than halved since the 1970s, falling from 68g of tea per person per week to only 25g. With a teabag or portion of loose tea weighing around 3g, that means Britons are drinking on average only eight cups of tea a week today, down from 23 cups in 1974. And, while tea remains the most drunk hot drink in the UK, households now spend more on coffee.  The data, published by the Department for Environment, Food and Rural Affairs as part of its open data scheme, is from 150,000 households who took part in the survey between 1974 and 2000, combined with information from 2000 to 2014. It shows some moves towards healthier diets in recent decades, with shifts to low-calorie soft drinks, from whole to skimmed milk and increasing consumption of fresh fruit. But, weekly consumption of chips, pizza, crisps and ready meals has soared.  There has also been a dramatic shift from white to brown, wholemeal and other bread but the figures suggest the amount people are eating has fallen from 25 to 15 slices a week over the past four decades, based on a 40g slice from a medium sliced loaf. The consumption of baked beans in sauce has dropped by a fifth despite a rise in other types of convenience food, particularly Italian dishes. Adults in the UK now eat an average of 75g of pizza every week compared with none in 1974, while the consumption of pasta has almost tripled over the same period.  Fresh potatoes are also becoming less essential with a 67 decrease from 1974, when adults ate the equivalent of 188g every day. Other vegetables such as cucumbers, courgettes, aubergines and mushrooms have gained space on the table. Consumption of takeaway food has almost doubled since 1974, from 80g per person per week to 150g. Around 33g of this amount is chips and 56g is meat, with kebabs (10g), chicken (7g), burgers (5g) and meat-based meals (32g) particularly popular.  Some trends suggest that British people are becoming more prudent in what they put on their plates, with the average consumption of fruit, both fresh and processed, increasing by 50 since 1974. In 2014, UK adults ate an average of 157g of fruit per day, equivalent to almost two portions of the five-a-day recommendation from the government. Bananas have been the most popular fruit in the UK since 1996, reaching 221g per adult per week in 2014, well above apples (131g) and oranges (48g). Low-calorie soft drinks represented half of all soft drinks consumed in 2014 for the first time.  Other social changes emerge from the survey, with questions about owning chickens and getting your own eggs being dropped in 1991 and a somewhat belated end in the same year to asking the housewife to fill out the questionnaire. Britons are spending a smaller proportion of pay on food today  11%, compared with 24 in 1974.  The UK Environment Secretary, Elizabeth Truss, said: Food is the heart and soul of our society and this data not only shows what we were eating 40 years ago but how a change in culture has led to a food revolution. Shoppers are more plugged in to where their food comes from than ever before, the internet has brought quality produce to our doorsteps at the click of a button, pop-up restaurants are showcasing the latest trends and exciting global cuisines are now as common as fish and chips.  By opening up this data, we can look beyond what, where or how previous generations were eating and pinpoint the moments that changed our habits for good. Weve only scraped the surface of what the National Food Survey can tell us and, from local food maps and school projects to predicting new food trends, I look forward to seeing how this data can be used to learn more about our past and grow our world-leading food and farming industry in the future.")