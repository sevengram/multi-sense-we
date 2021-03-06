# -*- coding:utf-8 -*-

import random

import numpy


def make_sampling_table(size, sampling_factor=1e-5):
    """
        This generates an array where the ith element
        is the probability that a word of rank i would be sampled,
        according to the sampling distribution used in word2vec.
        
        The word2vec formula is:
            p(word) = min(1, sqrt(word.frequency/sampling_factor) / (word.frequency/sampling_factor))

        We assume that the word frequencies follow Zipf's law (s=1) to derive 
        a numerical approximation of frequency(rank):
           frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))
        where gamma is the Euler-Mascheroni constant.
    """
    gamma = 0.577
    rank = numpy.array(list(range(size)))
    rank[0] = 1
    inv_fq = rank * (numpy.log(rank) + gamma) + 0.5 - 1. / (12. * rank)
    f = sampling_factor * inv_fq
    return numpy.minimum(1., f / numpy.sqrt(f))


def skipgrams(sequence, window_size=4, sampling_table=None, neg_sample_table=None, neg_sample_rate=1., shuffle=True):
    """ 
        Take a sequence (list of indexes of words), 
        returns couples of [word_index, other_word index] and labels (1s or 0s),
        where label = 1 if 'other_word' belongs to the context of 'word',
        and label=0 if 'other_word' is ramdomly sampled

        @param window_size: int. actually half-window. The window of a word wi will be [i-window_size, i+window_size+1]
        @param neg_sample_table: table used to create negative samples.
        @param neg_sample_rate: float >= 0. 0 for no negative (=random) samples. 1 for same number as positive samples.

        Note: by convention, index 0 in the vocabulary is a non-word and will be skipped.
    """
    couples = []
    labels = []
    seq_indices = []
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

        window_start = max(0, i - window_size)
        window_end = min(len(sequence), i + window_size + 1)
        for j in xrange(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                couples.append([wi, wj])
                seq_indices.append(i)
                labels.append(1)

    if neg_sample_rate > 0 and neg_sample_table:
        for i in xrange(int(len(labels) * neg_sample_rate)):
            word = couples[i % len(labels)][0]
            target = random.choice(neg_sample_table)
            if target == word:
                continue
            couples.append([word, target])
            seq_indices.append(seq_indices[i % len(labels)])
            labels.append(0)

    if shuffle:
        seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)
        random.seed(seed)
        random.shuffle(seq_indices)

    return couples, labels, seq_indices
