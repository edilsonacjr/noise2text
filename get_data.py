"""
    get_data.py

    First step in the noise2text pipeline.
    
    Author: Edilson A. Correa Junior <edilsonacjr@gmail.com>
"""

import os
import urllib.request
import json

import nltk
import numpy as np

from pathlib import Path
from zipfile import ZipFile

BASE_NILC_URL = 'http://143.107.183.175:22980/download.php?file=embeddings/'


class NILCDriver:
    def __init__(self, emb_type='word2vec', subtype='skip', dimension=100, output_path='embeddings',
                 output_pattern='{emb_type}_{subtype}_{dimension}', output_temp='emb_tmp', force_download=False):
        self.emb_type = emb_type
        self.subtype = subtype
        self.dimension = dimension
        self.file_name = output_pattern.format(emb_type=emb_type, subtype=subtype, dimension=dimension)
        self.output_save = os.path.join(output_path)
        self.output_tmp = os.path.join(output_path, output_temp, self.file_name)
        self.force_download = force_download

        if self.dimension not in [50, 100, 300, 600, 1000]:
            raise ValueError('Invalid dimension, try 50, 100, 300, 600 or 1000.')

        if self.emb_type == 'glove':
            self.URL = BASE_NILC_URL + '{}/{}_s{}.zip'.format(self.emb_type, self.emb_type, str(self.dimension))
        elif self.emb_type in ['word2vec', 'wang2vec', 'fasttext']:
            if self.subtype in ['cbow', 'skip']:
                self.URL = BASE_NILC_URL + '{}/{}_s{}.zip'.format(self.emb_type, self.subtype, str(self.dimension))
            else:
                raise ValueError('Invalid subtype, try cbow or skip.')
        else:
            raise ValueError('Invalid emb_type, try "word2vec", "glove", "wang2vec" or "fasttext".')

    def download(self):
        if not os.path.exists(self.output_tmp + '.zip') or self.force_download:
            response = urllib.request.urlopen(self.URL)
            data = response.read()
            if not os.path.exists(self.output_tmp):
                os.makedirs(self.output_tmp)
            with open(self.output_tmp + '.zip', 'wb') as emb_file:
                emb_file.write(data)
        else:
            print('Embedding already downloaded, if you want to download again please use force_download')

    def extract(self):
        with ZipFile(self.output_tmp + '.zip', 'r') as zipdata:
            # Extract all the contents of zip file in current directory
            zip_info = zipdata.infolist()[0]
            zip_info.filename = self.file_name + '.txt'
            zipdata.extract(zip_info, path=self.output_save)

    def download_extract(self):
        self.download()
        self.extract()


class NewsG1Driver:
    def __init__(self, file_name, path='data/', remove_stopwords=True, processed_pattern='processed_{file_name}'):
        self.file = os.path.join(path, file_name)
        self.file_save = os.path.join(path, processed_pattern.format(file_name=file_name))
        self.tokenizer = nltk.tokenize.RegexpTokenizer('(?u)\\b\\w\\w+\\b')
        if remove_stopwords:
            self.stopwords = nltk.corpus.stopwords.words('portuguese')
        else:
            self.stopwords = []
        self.sentences = []

    def preprocess(self):
        with open(self.file, 'r') as f_text:
            for line in f_text:
                original_tokens = self.tokenizer.tokenize(line)
                tokens = [w.lower() for w in original_tokens if w.lower() not in self.stopwords and
                          not w.isnumeric() and len(w) > 1]
                self.sentences.append(' '.join(tokens))

    def save_sentences(self):
        with open(self.file_save, 'w') as output_file:
            for sentence in self.sentences:
                output_file.write(sentence + '\n')


class MCIDriver:
    def __init__(self, file_name, path='data/', remove_stopwords=True, processed_pattern='processed_{file_name}'):
        self.file = os.path.join(path, file_name)
        self.file_save_setences = os.path.join(path, 'senteces_' + processed_pattern.format(file_name=file_name))
        self.file_save_targets = os.path.join(path, 'targets_' + processed_pattern.format(file_name=file_name))
        self.tokenizer = nltk.tokenize.RegexpTokenizer('(?u)\\b\\w\\w+\\b')
        if remove_stopwords:
            self.stopwords = nltk.corpus.stopwords.words('portuguese')
        else:
            self.stopwords = []
        self.sentences = []
        self.targets = []

    def preprocess(self):
        docs_CCL = []
        docs_NLS = []
        with open(self.file, 'r') as json_data:
            for line in json_data:
                doc = json.loads(line)
                if doc['group'] == 'MCI':
                    text = ' '.join(doc['sentences'])
                    original_tokens = self.tokenizer.tokenize(text)
                    tokens = [w.lower() for w in original_tokens if w.lower() not in self.stopwords and
                              not w.isnumeric() and len(w) > 1]
                    docs_CCL.append(' '.join(tokens))
                else:
                    text = ' '.join(doc['sentences'])
                    original_tokens = self.tokenizer.tokenize(text)
                    tokens = [w.lower() for w in original_tokens if w.lower() not in self.stopwords and
                              not w.isnumeric() and len(w) > 1]
                    docs_NLS.append(' '.join(tokens))

        self.targets = [1] * len(docs_CCL) + [0] * len(docs_NLS)
        self.sentences = docs_CCL + docs_CCL
        return self.sentences, self.targets

    def save_sentences(self):
        with open(self.file_save_setences, 'w') as output_file:
            for sentence in self.sentences:
                output_file.write(sentence + '\n')
        with open(self.file_save_targets, 'w') as output_file:
            for target in self.targets:
                output_file.write(str(target) + '\n')


def main():
    #nltk.download('stopwords')
    #driver = NILCDriver(emb_type='word2vec', subtype='cbow', dimension=50)
    #driver.download_extract()
    #g1 = NewsG1Driver(file_name='g1_final.txt')
    #g1.preprocess()
    #g1.save_sentences()

    mci = MCIDriver(file_name='data_set_cohmetrix_cn_trh_1.0.json')
    mci.preprocess()
    mci.save_sentences()


if __name__ == '__main__':
    main()
