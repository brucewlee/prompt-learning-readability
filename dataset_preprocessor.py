from glob import glob
from collections import defaultdict
from itertools import permutations

import pandas as pd

from utils import file_handler as f_handler

class GeneralPreprocessorForPairwiseInstances:
    def __init__(self, file_directories: list, corpus_type: str):
        self.file_directories = file_directories
        self.corpus_type = corpus_type

        self.current_file_directory = "updated in _iterate_file_directories"
        self.current_file_name = "updated in _iterate_file_directory"

        self.raw_corpus = []
        self.raw_corpus_by_slug = defaultdict(list)
        self.permutated_corpus = []

        self._build_raw_corpus()
        self._permutate_raw_corpus()
        self._report_statistics()

    def _build_raw_corpus(self):
        self._iterate_file_directories()

    def _iterate_file_directories(self):
        for file_directory in self.file_directories:
            self.current_file_directory = file_directory
            self._iterate_file_directory()

    def _iterate_file_directory(self):
        for file_name in glob(f"{self.current_file_directory}*"):
            self.current_file_name = file_name
            self._read_file_and_append_raw_corpus()

    def _read_file_and_append_raw_corpus(self):
        return NotImplementedError

    def _map_label(self):
        return NotImplementedError

    def _permutate_raw_corpus(self):
        if self.corpus_type == "parallel":
            self._rebuild_raw_corpus_by_slug()
            self._permutate_parallel_and_relabel()
        elif self.corpus_type == "distinct":
            self._permutate_distinct_and_relabel()

    def _rebuild_raw_corpus_by_slug(self):
        for instance in self.raw_corpus:
            self.raw_corpus_by_slug[instance['slug_name']].append(instance)
    
    def _permutate_parallel_and_relabel(self):
        for slug in self.raw_corpus_by_slug.values():
            perms = permutations(slug, 2) 
            for paired_instance in list(perms):
                if paired_instance[0]['label'] > paired_instance[1]['label']:
                    difficult_text = 'text'
                elif paired_instance[0]['label'] < paired_instance[1]['label']:
                    difficult_text = 'text_pair'
                self.permutated_corpus.append(
                    {
                        'slug_name': paired_instance[0]['slug_name'],
                        'text': paired_instance[0]['text'],
                        'text_pair': paired_instance[1]['text'],
                        'text_label': paired_instance[0]['label'],
                        'text_pair_label': paired_instance[1]['label'],
                        'difficult_text': difficult_text
                    }
                )

    def _permutate_distinct_and_relabel(self):
        perms = permutations(self.raw_corpus, 2)
        for paired_instance in list(perms):
            if paired_instance[0]['label'] > paired_instance[1]['label']:
                difficult_text = 'text'
            elif paired_instance[0]['label'] < paired_instance[1]['label']:
                difficult_text = 'text_pair'
            else:
                # skip paired_instances that have two same labels
                continue
            self.permutated_corpus.append(
                {
                    'text': paired_instance[0]['text'],
                    'text_pair': paired_instance[1]['text'],
                    'text_label': paired_instance[0]['label'],
                    'text_pair_label': paired_instance[1]['label'],
                    'difficult_text': difficult_text
                }
            )

    def _preprocess_text(self, text):
        text = text.replace("\n", " ")
        text = text.replace("##", "")
        text = text.replace("Intermediate","")
        text = text.encode('ascii', errors='ignore').decode()
        text = " ".join(text.split())
        return text

    def _report_statistics(self):
        print(f"{type(self).__name__}: There is a total number of {len(self.raw_corpus)} files (or texts) in {self.file_directories}.\n")
        print(f"{type(self).__name__}: There is a total number of {len(self.raw_corpus_by_slug)} slugs in {self.file_directories}.\n")
        print(f"{type(self).__name__}: After permutation, there is a total number of {len(self.permutated_corpus)} pairwise instances in {self.file_directories}.\n")

    def save_csv(self, path: str, split: tuple = None, random_seed: int = 2022):
        df = pd.DataFrame(self.permutated_corpus)
        f_handler.get_pandas_and_save_ndjson(df, path + '.json')

        if split != None:
            assert split[0] + split[1] + split[2] == 1, "check train/dev/test ratio"

            ratio_train = split[0]
            train_df = df.sample(frac = ratio_train, random_state = random_seed)
            dev_test_df = df.drop(train_df.index)
            
            ratio_dev = split[1]/(1 - split[0])
            dev_df = dev_test_df.sample(frac = ratio_dev, random_state = random_seed)
            test_df = dev_test_df.drop(dev_df.index)

            print(f"{type(self).__name__}: created splits of train - {len(train_df)}, dev - {len(dev_df)}, test - {len(test_df)},")

            f_handler.get_pandas_and_save_ndjson(train_df, path + '_train' + '.json')
            f_handler.get_pandas_and_save_ndjson(dev_df, path + '_dev' + '.json')
            f_handler.get_pandas_and_save_ndjson(test_df, path + '_test' + '.json')

class OneStopEnglishPreprocessorForPairwiseInstances(
    GeneralPreprocessorForPairwiseInstances
    ):
    def _read_file_and_append_raw_corpus(self):
        slug_name = self.current_file_name.replace(self.current_file_directory,'')[:-8]
        label = self._map_label(self.current_file_directory[-8:-5])
        with open (self.current_file_name, 'r') as file:
            text = file.read()
            text = self._preprocess_text(text)
        self.raw_corpus.append(
            {
                'slug_name': slug_name,
                'label': label, 
                'text': text
            }
        )
    
    def _map_label(self, label):
        mapper = {
            "Adv": 2,
            "Int": 1,
            "Ele": 0
        }
        return mapper[label]



class NewselaPreprocessorForPairwiseInstances(
    GeneralPreprocessorForPairwiseInstances
    ):
    def _read_file_and_append_raw_corpus(self):
        slug_name = self.current_file_name.replace(self.current_file_directory,'')[:-9]
        label = self._map_label(self.current_file_name[-5])
        with open (self.current_file_name, 'r') as file:
            text = file.read()
            text = self._preprocess_text(text)
        self.raw_corpus.append(
            {
                'slug_name': slug_name,
                'label': label, 
                'text': text
            }
        )

    def _map_label(self, label):
        mapper = {
            "0": 5,
            "1": 4,
            "2": 3,
            "3": 2,
            "4": 1,
            "5": 0
        }
        return mapper[label]



class CommonCoreStandardsPreprocessorForPairwiseInstances(
    GeneralPreprocessorForPairwiseInstances
    ):
    def _build_raw_corpus(self):
        for file_directory in self.file_directories:
            df = pd.read_csv(file_directory)
            df = df[['Class', 'Text']]
            df.columns = ['label', 'text']
            df = self._map_label(df)
            self.raw_corpus.extend(df.to_dict('records'))
    
    def _map_label(self, df):
        mapper = {
            "F": 5,
            "E": 4,
            "D": 3,
            "C": 2,
            "B": 1,
            "A": 0
        }
        df['label'] = df['label'].map(mapper)
        return df



class CambridgeEnglishReadabilityPreprocessorForPairwiseInstances(
    GeneralPreprocessorForPairwiseInstances
    ):
    def _read_file_and_append_raw_corpus(self):
        label = self._map_label(self.current_file_directory[-4:-1])
        with open (self.current_file_name, 'r') as file:
            text = file.read()
            text = self._preprocess_text(text)
        self.raw_corpus.append(
            {
                'label': label, 
                'text': text
            }
        )

    def _map_label(self, label):
        mapper = {
            "CPE": 4,
            "CAE": 3,
            "FCE": 2,
            "PET": 1,
            "KET": 0
        }
        return mapper[label]

if __name__ == "__main__":
    NewselaPreprocessor = NewselaPreprocessorForPairwiseInstances(
        file_directories = [
            'datasets/newsela_article_corpus_2016-01-29/articles/'
            ],
        corpus_type = "parallel"
        )
    NewselaPreprocessor.save_csv(
        'datasets/final_NEWS',
        split = (0.6,0.2,0.2)
        )

    OneStopEnglishPreprocessor = OneStopEnglishPreprocessorForPairwiseInstances(
        file_directories = [
            'datasets/OneStopEnglish/Texts-SeparatedByReadingLevel/Adv-Txt/', 
            'datasets/OneStopEnglish/Texts-SeparatedByReadingLevel/Int-Txt/', 
            'datasets/OneStopEnglish/Texts-SeparatedByReadingLevel/Ele-Txt/',
            ],
        corpus_type = "parallel"
        )
    OneStopEnglishPreprocessor.save_csv(
        'datasets/final_OSEN',
        split = (0.6,0.2,0.2)
        )

    CommonCoreStandardsPreprocessor = CommonCoreStandardsPreprocessorForPairwiseInstances(
        file_directories = [
            'datasets/CommonCoreStandards/Story.csv'
            ],
        corpus_type = "distinct"
        )
    CommonCoreStandardsPreprocessor.save_csv('datasets/final_CCSB')

    CambridgeEnglishReadabilityPreprocessor = CambridgeEnglishReadabilityPreprocessorForPairwiseInstances(
        file_directories = [
            'datasets/CambridgeEnglishReadability/Readability_dataset/CAE/',
            'datasets/CambridgeEnglishReadability/Readability_dataset/CPE/',
            'datasets/CambridgeEnglishReadability/Readability_dataset/FCE/',
            'datasets/CambridgeEnglishReadability/Readability_dataset/KET/',
            'datasets/CambridgeEnglishReadability/Readability_dataset/PET/',
            ],
        corpus_type = "distinct"
        )
    CambridgeEnglishReadabilityPreprocessor.save_csv('datasets/final_CAMB')