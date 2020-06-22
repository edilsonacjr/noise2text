"""
    Autoencoders
    
    Author: Edilson A. Correa Junior <edilsonacjr@gmail.com>
"""
import os
import numpy as np
from pathlib import Path

from keras.layers import Embedding, Input, LSTM, RepeatVector, Dense, Dropout, GaussianNoise
from keras.models import Model
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

from gensim.models import KeyedVectors

# MacOS problem
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Autoencoder():

    def __init__(self,
                 embedding_dim=50,
                 encoding_dim=300,
                 embedding_file='word2vec_cbow_50.txt',
                 embedding_path='embeddings/',
                 bidirectional=False,
                 optimizer='nadam',
                 loss='binary_crossentropy',
                 metrics=['binary_crossentropy'],
                 checkpoint=True,
                 cp_monitor='val_loss',
                 cp_folder='./checkpoint',
                 cp_filename_prefix='chkp_{epoch:02d}-{val_loss:.2f}.hdf5',
                 cp_save_best_only=True,
                 cp_save_period=10,
                 max_num_words=500,
                 max_sequence_length=100,
                 noise=0.2):

        self.embedding_dim = embedding_dim
        self.encoding_dim = encoding_dim
        self.embedding_filepath = Path(embedding_path) / embedding_file
        self.bidirectional = bidirectional
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.checkpoint = checkpoint
        self.cp_monitor = cp_monitor
        self.cp_folder = cp_folder
#        self.cp_folder.mkdir(parents=True, exist_ok=True)
        self.cp_filename_prefix = cp_filename_prefix
        self.cp_save_best_only = cp_save_best_only
        self.cp_save_period = cp_save_period
        self.max_num_words = max_num_words
        self.max_sequence_length = max_sequence_length
        self.noise = noise
        self.model = None
        self.encoder = None
        self.noise_encoder = None

        if self.cp_monitor not in ['val_loss', 'val_acc']:
            raise ValueError('Invalid cp_monitor, try "val_loss" or "val_acc"')

    def data_formatting(self, X, max_num_docs=100):
        tokenizer = Tokenizer(num_words=self.max_num_words+1)
        tokenizer.fit_on_texts(X)
        # Bug on Tokenizer when creating the index, it changed the way that worked on old versions
        tokenizer.word_index = {e: i for e, i in tokenizer.word_index.items() if i <= self.max_num_words}
        tokenizer.word_index[tokenizer.oov_token] = self.max_num_words + 1
        ##

        sequences = tokenizer.texts_to_sequences(X)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        x_train = pad_sequences(sequences[:max_num_docs], maxlen=self.max_sequence_length, padding='pre',
                                truncating='pre')
        y_train = tokenizer.texts_to_matrix(X[:max_num_docs], mode='binary')

        print('Shape of data tensor:', x_train.shape)

        return x_train, y_train, word_index, y_train.shape[1]

    def build(self, word_index, output_shape):

        # Loading Word2Vec
        model = KeyedVectors.load_word2vec_format(self.embedding_filepath, binary=False)

        embedding_matrix = np.zeros((len(word_index) + 1, self.embedding_dim))
        for word, i in word_index.items():
            if word in model:
                embedding_matrix[i] = model[word]
            else:
                embedding_matrix[i] = np.random.rand(1, self.embedding_dim)[0]

        embedding_layer = Embedding(len(word_index) + 1,
                                    self.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_sequence_length,
                                    trainable=True)

        inputs = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(inputs)
        encoded = LSTM(self.encoding_dim, name='encoder_layer')(embedded_sequences)
        noise = GaussianNoise(self.noise, name='noise_encoder_layer')(encoded)

        decoded = RepeatVector(self.max_sequence_length)(noise)
        decoded = LSTM(self.encoding_dim)(decoded)

        # decoded = Dropout(0.5)(decoded)
        decoded = Dense(output_shape, activation='softmax')(decoded)

        self.model = Model(inputs, decoded)

        self.encoder = Model(inputs, encoded)
        self.noise_encoder = Model(inputs, noise)

    def _compile(self):
        print('Compiling...')
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def _summary(self):
        self.model.summary()

    def save_model(self, filename='model', path='model'):
        # TODO
        # path
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("%s.json" % filename, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("%s.h5" % filename)
        print("Saved model to disk")

    def load_model(self, filename='model', path='model'):
        # TODO
        # path
        json_file = open("%s.json" % filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model .load_weights("%s.h5" % filename)
        self.encoder = Model(self.model.input, self.model.get_layer('encoder_layer').output)
        self.noise_encoder = Model(self.model.input, self.model.get_layer('noise_encoder_layer').output)

    def save_model_single_file(self, filename='model.h5', path='model'):
        model_path = Path(path)
        model_path.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path / filename)

    def load_model_single_file(self, filename='model.h5', path='model'):
        model_path = Path(path)
        self.model = load_model(model_path / filename)
        self.encoder = Model(self.model.input, self.model.get_layer('encoder_layer').output)
        self.noise_encoder = Model(self.model.input, self.model.get_layer('noise_encoder_layer').output)

    def fit(self, X, y, epochs, batch_size, stopped=False, shuffle=True):
        self._compile()
        self._summary()

        callbacks_list = []
        if self.checkpoint:
            checkpoint = ModelCheckpoint(self.cp_folder + '/' +  self.cp_filename_prefix,
                                         monitor=self.cp_monitor,
                                         verbose=1,
                                         save_best_only=self.cp_save_best_only,
                                         mode='auto')
            callbacks_list.append(checkpoint)

        if stopped:
            self.model.load_weights(self.cp_folder / 'weights.best.hdf5')

        self.model.fit(X, y,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       validation_split=0.1,
                       callbacks=callbacks_list)


def main():
    rnn_ae = Autoencoder()
    with open('data/processed_g1_final.txt') as g1_file:
        data = [line for line in g1_file]

    x_train, y_train, word_index, output_shape = rnn_ae.data_formatting(data)

    rnn_ae.build(word_index, output_shape)
    rnn_ae.fit(x_train, y_train, 10, 128)
    rnn_ae.save_model_single_file()


if __name__ == '__main__':
    main()
