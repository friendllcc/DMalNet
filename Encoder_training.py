#!/usr/bin/env python
# coding: utf-8

# In[4]:

import numpy as np

# Word2Vec Encoder
from gensim.models import Word2Vec

# Similarity Encoder
from dirty_cat import SimilarityEncoder

import pickle

# In[3]:

''' word2vec training '''


def word2vec_training():
    # load training corpus
    api_sequence = np.load('./api_sequence_for_word2vec_encoder.npz', allow_pickle=True)
    api_sequence = api_sequence['api_sequence']

    # sg=0 CBOW
    # sg=1 skip-gram
    model = Word2Vec(api_sequence, sg=1, size=32, window=10, min_count=3, negative=3, sample=0.001, hs=1, workers=8)
    model.save('./encoder/skip-gram_previous.model')


# In[16]:

''' similarity encoders training '''

def similarity_encoder_training():
    # load training corpus
    data = np.load('./data_for_similarity_encoder.npz', allow_pickle=True)
    paths_list = data['paths_list']
    dlls_list = data['dlls_list']
    urls_list = data['urls_list']
    registry_list = data['registry_list']
    ips_list = data['ips_list']

    # paths similarity encoder

    paths_sim = SimilarityEncoder(ngram_range=(3, 5), random_state=10, n_prototypes=16, categories='most_frequent')

    paths_sim.fit(np.asarray(paths_list).reshape([-1, 1]))

    output_hal = open("./encoder/paths_sim.pkl", 'wb')
    str_paths_sim = pickle.dumps(paths_sim)
    output_hal.write(str_paths_sim)
    output_hal.close()

    # dlls similarity encoder

    dlls_sim = SimilarityEncoder(ngram_range=(3, 5), random_state=10, n_prototypes=16, categories='most_frequent')

    dlls_sim.fit(np.asarray(dlls_list).reshape([-1, 1]))

    output_hal = open("./encoder/dlls_sim.pkl", 'wb')
    str_dlls_sim = pickle.dumps(dlls_sim)
    output_hal.write(str_dlls_sim)
    output_hal.close()

    # urls similarity encoder

    urls_sim = SimilarityEncoder(ngram_range=(3, 5), random_state=10, n_prototypes=16, categories='most_frequent')

    urls_sim.fit(np.asarray(urls_list).reshape([-1, 1]))

    output_hal = open("./encoder/urls_sim.pkl", 'wb')
    str_urls_sim = pickle.dumps(urls_sim)
    output_hal.write(str_urls_sim)
    output_hal.close()

    # registry similarity encoder

    registry_sim = SimilarityEncoder(ngram_range=(3, 5), random_state=10, n_prototypes=16, categories='most_frequent')

    registry_sim.fit(np.asarray(registry_list).reshape([-1, 1]))

    output_hal = open("./encoder/registry_sim.pkl", 'wb')
    str_registry_sim = pickle.dumps(registry_sim)
    output_hal.write(str_registry_sim)
    output_hal.close()

    # ips similarity encoder

    ips_sim = SimilarityEncoder(ngram_range=(3, 5), random_state=10, n_prototypes=16, categories='most_frequent')

    ips_sim.fit(np.asarray(ips_list).reshape([-1, 1]))

    output_hal = open("./encoder/ips_sim.pkl", 'wb')
    str_ips_sim = pickle.dumps(ips_sim)
    output_hal.write(str_ips_sim)
    output_hal.close()
