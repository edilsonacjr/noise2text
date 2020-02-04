"""
    Autoencoders
    
    Author: Edilson A. Correa Junior <edilsonacjr@gmail.com>
"""

from keras.callbacks import ModelCheckpoint


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
                 cp_save_best_only= True,
                 cp_save_period=10,):

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
        self.model = None

        if self.cp_monitor not in ['val_loss', 'val_acc']:
            raise ValueError('Invalid cp_monitor, try "val_loss" or "val_acc"')

    def build(self):
        pass

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

    def load_model(self):
        pass

    def fit(self, X, y, epochs, batch_size, stopped=False):
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

        self.model.fit(X_input_dic, y,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       validation_split=0.1,
                       callbacks=callbacks_list)




