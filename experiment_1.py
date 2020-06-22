"""
    Brief description
    
    Author: Edilson A. Correa Junior <edilsonacjr@gmail.com>
"""

import json
import nltk
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from get_data import NewsG1Driver, NILCDriver, MCIDriver
from autoencoder import Autoencoder


def main():

    # Data loading and preprocessing
    nltk.download('stopwords')
    driver = NILCDriver(emb_type='word2vec', subtype='cbow', dimension=50)
    driver.download_extract()
    g1 = NewsG1Driver(file_name='g1_final.txt')
    g1.preprocess()
    g1.save_sentences()

    # MCI data
    mci = MCIDriver(file_name='data_set_cohmetrix_cn_trh_1.0.json')
    mci_data, mci_target = mci.preprocess()

    with open('data/processed_g1_final.txt') as g1_file:
        data = [line for line in g1_file]

    # Model training
    rnn_ae = Autoencoder()
    x_train, y_train, word_index, output_shape = rnn_ae.data_formatting(data)

    x_test, _, _, _ = rnn_ae.data_formatting(mci_data)

    rnn_ae.build(word_index, output_shape)
    rnn_ae.fit(x_train, y_train, 10, 128)
    rnn_ae.save_model_single_file()

    x_train_clean = rnn_ae.encoder.predict(x_train)
    x_train_noise = rnn_ae.noise_encoder.predict(x_train)
    x_train_encoded = np.concatenate([x_train_clean, x_train_noise])
    y_train_encoded = [0] * len(x_train_clean) + [1] * len(x_train_noise)

    x_test_encoded = rnn_ae.encoder.predict(x_test)
    y_test_encoded = mci_target

    clf = LogisticRegression()
    clf.fit(x_train_encoded, y_train_encoded)

    print("Accuracy: %0.2f" % accuracy_score(y_test_encoded, clf.predict(x_test_encoded)))


if __name__ == '__main__':
    main()
