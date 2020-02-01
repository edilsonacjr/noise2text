"""
    get_data.py

    First step in the noise2text pipeline.
    
    Author: Edilson A. Correa Junior <edilsonacjr@gmail.com>
"""

import urllib.request

BASE_NILC_URL = 'http://143.107.183.175:22980/download.php?file=embeddings/'


class NILCDriver:
    def __init__(self, type='Word2Vec', subtype='', dimension=100, output='embeddings.zip'):
        self.type = type
        self.subtype = subtype
        self.dimension = dimension
        self.output = output

        if self.dimension not in [50, 100, 300, 600, 1000]:
            raise ValueError('Invalid dimension, try 50, 100, 300, 600 or 1000.')

        if self.type == 'glove':
            self.URL = BASE_NILC_URL + '{}/{}_s{}.zip'.format(self.type, self.type, str(self.dimension))
        elif self.type in ['word2vec', 'wang2vec', 'fasttext']:
            if self.subtype in ['cbow', 'skip']:
                self.URL = BASE_NILC_URL + '{}/{}_s{}.zip'.format(self.type, self.subtype, str(self.dimension))
            else:
                raise ValueError('Invalid subtype, try cbow or skip.')
        else:
            raise ValueError('Invalid type, try "word2vec", "glove", "wang2vec" or "fasttext".')

    def download(self):
        response = urllib.request.urlopen(self.URL)
        data = response.read()
        with open(self.output, 'wb') as emb_file:
            emb_file.write(data)

    def format(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()