from glob import glob
from collections import defaultdict
from itertools import permutations

import pandas as pd

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

    def _preprocess_text(self, text):
        text = text.replace("\n", " ")
        text = text.replace("##", "")
        text = text.encode('ascii', errors='ignore').decode()
        
        text = " ".join(text.split())
        return text

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
        for slug_label_text in self.raw_corpus:
            self.raw_corpus_by_slug[slug_label_text[0]].append(
                {
                    'text': slug_label_text[2], 
                    'label': slug_label_text[1],
                    'slug_name': slug_label_text[0]
                }
            )
    
    def _permutate_parallel_and_relabel(self):
        for slug in self.raw_corpus_by_slug.values():
            perms = permutations(slug, 2) 
            for slug_slug in list(perms):
                if slug_slug[0]['label'] > slug_slug[1]['label']:
                    difficult_text = 'text'
                elif slug_slug[0]['label'] < slug_slug[1]['label']:
                    difficult_text = 'text_pair'
                self.permutated_corpus.append(
                    {
                        'slug_name': slug[0]['slug_name'],
                        'text': slug[0]['text'],
                        'text_pair': slug[1]['text'],
                        'text_label': slug[0]['label'],
                        'text_pair_label': slug[1]['label'],
                        'difficult_text': difficult_text
                    }
                )

    def _permutate_distinct_and_relabel(self):
        return NotImplementedError

    def _report_statistics(self):
        print(f"{type(self).__name__}: There is a total number of {len(self.raw_corpus)} files (or texts) in {self.file_directories}.\n")
        print(f"{type(self).__name__}: There is a total number of {len(self.raw_corpus_by_slug)} slugs in {self.file_directories}.\n")
        print(f"{type(self).__name__}: After permutation, there is a total number of {len(self.permutated_corpus)} pairwise instances in {self.file_directories}.\n")

    def save_csv(self, path: str):
        df = pd.DataFrame(self.permutated_corpus)
        df.to_csv(path, index = False)


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
            (slug_name, label, text)
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
            (slug_name, label, text)
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



if __name__ == "__main__":
    NewselaPreprocessor = NewselaPreprocessorForPairwiseInstances(
        file_directories = [
            'datasets/newsela_article_corpus_2016-01-29/articles/'
            ],
        corpus_type = "parallel"
        )
    NewselaPreprocessor.save_csv('datasets/final_NEWS.csv')

    OneStopEnglishPreprocessor = OneStopEnglishPreprocessorForPairwiseInstances(
        file_directories = [
            'datasets/OneStopEnglish/Texts-SeparatedByReadingLevel/Adv-Txt/', 
            'datasets/OneStopEnglish/Texts-SeparatedByReadingLevel/Int-Txt/', 
            'datasets/OneStopEnglish/Texts-SeparatedByReadingLevel/Ele-Txt/',
            ],
        corpus_type = "parallel"
        )
    OneStopEnglishPreprocessor.save_csv('datasets/final_OSEN.csv')