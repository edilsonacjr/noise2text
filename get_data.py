"""
    get_data.py

    First step in the noise2text pipeline.
    
    Author: Edilson A. Correa Junior <edilsonacjr@gmail.com>
"""

import os
import urllib.request

from pathlib import Path
from zipfile import ZipFile

BASE_NILC_URL = 'http://143.107.183.175:22980/download.php?file=embeddings/'


class NILCDriver:
    def __init__(self, emb_type='word2vec', subtype='skip', dimension=100, output_path='embeddings',
                 output_pattern='/{emb_type}_{subtype}_{dimension}', output_temp='emb_tmp', force_download=False):
        self.emb_type = emb_type
        self.subtype = subtype
        self.dimension = dimension
        self.file_name = output_pattern.format(emb_type=emb_type, subtype=subtype, dimension=dimension)
        self.output_save = os.path.join(output_path)
        self.output_tmp = os.path.join(output_path, output_temp)
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
        if not os.path.exists(self.output_tmp + self.file_name + '.zip') or self.force_download:
            response = urllib.request.urlopen(self.URL)
            data = response.read()
            if not os.path.exists(self.output_tmp):
                os.makedirs(self.output_tmp)
            with open(self.output_tmp + self.file_name + '.zip', 'wb') as emb_file:
                emb_file.write(data)
        else:
            print('Embedding already downloaded, if you want to download again please use force_download')

    def extract(self):
        with ZipFile(self.output_tmp + self.file_name + '.zip', 'r') as zipdata:
            # Extract all the contents of zip file in current directory
            zip_info = zipdata.infolist()[0]
            zip_info.filename = self.file_name + '.txt'
            zipdata.extract(zip_info, path=self.output_save)

    def download_extract(self):
        self.download()
        self.extract()


def main():
    pass


if __name__ == '__main__':
    main()