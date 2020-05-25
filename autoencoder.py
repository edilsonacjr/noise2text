"""
    Autoencoders
    
    Author: Edilson A. Correa Junior <edilsonacjr@gmail.com>
"""

import numpy as np

from keras.layers import Embedding, Input, LSTM, RepeatVector, Dense, Dropout, GaussianNoise
from keras.models import Model
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

from gensim.models import KeyedVectors


class Autoencoder():

    def __init__(self,
                 embedding_dim=100,
                 encoding_dim=300,
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
                 max_num_words=150000,
                 max_sequence_length=100,
                 noise=0.2):

        self.embedding_dim = embedding_dim
        self.encoding_dim = encoding_dim
        self.bidirectional = bidirectional
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.checkpoint = checkpoint
        self.cp_monitor = cp_monitor
        self.cp_folder = cp_folder
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

    def data_formatting(self, X, max_num_docs):
        tokenizer = Tokenizer(num_words=self.max_num_words)
        tokenizer.fit_on_texts(X)
        sequences = tokenizer.texts_to_sequences(X)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        print(word_index.items())

        x_train = pad_sequences(sequences[:max_num_docs], maxlen=self.max_sequence_length, padding='pre',
                                truncating='pre')
        y_train = tokenizer.texts_to_matrix(X[:max_num_docs], mode='binary')

        print('Shape of data tensor:', x_train.shape)

        return x_train, y_train, word_index, y_train.shape[1]

    def build(self, word_index, output_shape):

        # Loading Word2Vec
        model = KeyedVectors.load_word2vec_format('glove_s300.bin', binary=True)

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

    def save_model(self, filename):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("%s.json" % filename, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("%s.h5" % filename)
        print("Saved model to disk")

    def load_model(self, filename):
        json_file = open("%s.json" % filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model .load_weights("%s.h5" % filename)
        self.encoder = Model(self.model.input, self.model.get_layer('encoder_layer').output)
        self.noise_encoder = Model(self.model.input, self.model.get_layer('noise_encoder_layer').output)

    def save_model_single_file(self, filename):
        self.model.save("model.h5")

    def load_model_single_file(self):
        self.model = load_model('model.h5')
        self.encoder = Model(self.model.input, self.model.get_layer('encoder_layer').output)
        self.noise_encoder = Model(self.model.input, self.model.get_layer('noise_encoder_layer').output)

    def fit(self, X, y, epochs, batch_size, stopped=False, shuffle=True):
        callbacks_list = []
        if self.checkpoint:
            checkpoint = ModelCheckpoint(self.cp_folder,
                                         monitor=self.cp_monitor,
                                         verbose=1,
                                         save_best_only=self.cp_save_best_only,
                                         mode='auto')
            callbacks_list.append(checkpoint)

        if stopped:
            self.model.load_weights(self.cp_filename)

        self.model.fit(X, y,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       validation_split=0.1,
                       callbacks=callbacks_list)


def main():
    pass


if __name__ == '__main__':
    main()