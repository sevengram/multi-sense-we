# -*- coding:utf-8 -*-

import cPickle
import collections

import numpy
from numpy import random, zeros, var
from theano import tensor as T
from scipy.spatial import distance

import sequence
import text
from trainers import SGD, AdaGrad
from user import UserClassifier

dist_func = {
    'COS': distance.cosine,
    'EUC': distance.euclidean
}


def nearest_k_points(target, points, k, dist_type):
    d = numpy.asarray(dist_func[dist_type](target, p) for p in points)
    return {i: d[i] for i in numpy.argpartition(d, k)[:k]}


class WordEmbeddingModel(object):
    def __init__(self, words_limit=5000, dimension=128, min_count=5):
        self.words_limit = words_limit
        self.dimension = dimension
        self.tokenizer = None
        self.word_list = []
        self.word_matrix_index = {}
        self.wordvec_matrix = None
        self.weight_matrix = None
        self.biases = None
        self.min_count = min_count

    def _init_values(self):
        self.wordvec_matrix = (random.randn(self.words_limit, self.dimension).astype(
            numpy.float32) - 0.5) / self.dimension
        self.weight_matrix = zeros((self.words_limit, self.dimension), dtype=numpy.float32)
        self.biases = zeros(self.words_limit, dtype=numpy.float32)

    def build_vocab(self, texts, *args, **kwargs):
        self.tokenizer = text.Tokenizer()
        self.tokenizer.fit_on_texts(texts, words_limit=self.words_limit, min_count=self.min_count)
        self._load_words(*args, **kwargs)

    def load_vocab(self, path, *args, **kwargs):
        self.tokenizer = cPickle.load(open(path, 'rb'))
        self._load_words(*args, **kwargs)

    def _load_words(self, *args, **kwargs):
        self.words_limit = min(self.words_limit, self.tokenizer.count())
        self.word_list = self.tokenizer.word_list
        for i in range(len(self.word_list)):
            self.word_matrix_index[self.word_list[i]] = [i]
        self._init_values()

    def dump(self, path):
        if path:
            params = [self.wordvec_matrix,
                      self.weight_matrix,
                      self.biases,
                      self.word_matrix_index,
                      self.word_list]
            cPickle.dump(params, open(path, "wb"))

    def load(self, path):
        params = cPickle.load(open(path, "rb"))
        self.wordvec_matrix = params[0]
        self.weight_matrix = params[1]
        self.biases = params[2]

    def load_word_vectors(self, path):
        self.wordvec_matrix = cPickle.load(open(path, "rb"))

    def save_tokenizer(self, path):
        if path:
            cPickle.dump(self.tokenizer, open(path, "wb"))

    def save_word_list(self, path):
        if path:
            cPickle.dump(self.word_list, open(path, "wb"))

    def save_word_index(self, path):
        if path:
            cPickle.dump(self.word_matrix_index, open(path, "wb"))

    def save_word_vectors(self, path):
        if path:
            cPickle.dump(self.wordvec_matrix, open(path, "wb"))

    def save_weight_matrix(self, path):
        if path:
            cPickle.dump(self.weight_matrix, open(path, "wb"))

    def fit(self, texts, nb_epoch=1, monitor=None, **kwargs):
        raise NotImplementedError()

    def _sequentialize(self, texts, **kwargs):
        raise NotImplementedError()

    def nearest_words(self, word, limit=10, dist_type='COS'):
        if self.wordvec_matrix is None:
            print('load vocab and model first!')
            return None
        senses = self.word_matrix_index.get(word)
        result = []
        for si in senses:
            if si is None or si >= self.wordvec_matrix.shape[0]:
                print('can\'t find this word!')
                return None
            else:
                result.append({self.word_list[i]: v for i, v in
                               nearest_k_points(self.wordvec_matrix[si], self.wordvec_matrix, limit,
                                                dist_type).iteritems()})
        return result

    def get_senses(self, wi):
        return self.word_matrix_index[self.word_list[wi]]

    def get_words_text(self, word_indices):
        return [self.word_list[i] for i in word_indices]


class SkipGramNegSampEmbeddingModel(WordEmbeddingModel):
    def __init__(self, words_limit=5000, dimension=128, window_size=5, neg_sample_rate=1., batch_size=8, min_count=5):
        super(SkipGramNegSampEmbeddingModel, self).__init__(words_limit=words_limit, dimension=dimension,
                                                            min_count=min_count)
        self.window_size = window_size
        self.batch_size = batch_size
        self.neg_sample_rate = neg_sample_rate
        self.trainer = None

    def _sequentialize(self, texts, sampling=True, **kwargs):
        sampling_table = sequence.make_sampling_table(self.words_limit) if sampling else None
        for seq in self.tokenizer.texts_to_sequences_generator(texts):
            yield seq, sequence.skipgrams(seq, window_size=self.window_size,
                                          sampling_table=sampling_table,
                                          neg_sample_table=self.tokenizer.unigram_table,
                                          neg_sample_rate=self.neg_sample_rate)

    def set_trainer(self, lr=.1, optimizer='sgd', **kwargs):
        x = T.fmatrix("x")
        w = T.fmatrix("w")
        b = T.fvector("b")
        y = T.bvector("y")
        hx = 1 / (1 + T.exp(-T.sum(w * x, axis=1) - b))
        obj = y * T.log(hx) + (1 - y) * T.log(1 - hx)

        if optimizer == 'sgd':
            self.trainer = SGD(lr=lr,
                               lr_b=kwargs.get('lr_b'),
                               momentum=kwargs.get('momentum', 0.0),
                               momentum_b=kwargs.get('momentum_b'))
        elif optimizer == 'adagrad':
            self.trainer = AdaGrad(lr=lr,
                                   lr_b=kwargs.get('lr_b'),
                                   epsilon=kwargs.get('epsilon', 1e-6),
                                   gx_shape=(self.batch_size, self.dimension),
                                   gw_shape=(self.batch_size, self.dimension),
                                   gb_shape=self.batch_size)
        else:
            raise NotImplementedError()
        self.trainer.compile(x, w, b, y, obj)

    def fit(self, texts, nb_epoch=1, sampling=True, monitor=None):
        for e in range(nb_epoch):
            for k, (seq, (couples, labels, seq_indices)) in enumerate(self._sequentialize(texts, sampling)):
                for i in range(0, len(couples) - self.batch_size, self.batch_size):
                    wi, wj = numpy.array(zip(*couples[i:i + self.batch_size]))
                    self.trainer.update(self.wordvec_matrix, self.weight_matrix, self.biases,
                                        labels[i:i + self.batch_size], wi, wj)

    def context_words_indices(self, seq, si, with_si=False):
        return seq[max(0, si - self.window_size): si] + [si] if with_si else [] + seq[(si + 1):si + self.window_size]

    def context_text(self, seq, si):
        return ' '.join([self.word_list[i] for i in self.context_words_indices(seq, si, True)])


class ClusteringSgNsEmbeddingModel(SkipGramNegSampEmbeddingModel):
    def __init__(self, words_limit=5000, dimension=128, window_size=5, neg_sample_rate=1., batch_size=8, sense_limit=5,
                 threshold=.5, min_count=5, distance_type='COS', use_dpmeans=True, multi_sense_word_limit=None,
                 single_sense_list=None):
        super(ClusteringSgNsEmbeddingModel, self).__init__(words_limit=words_limit, dimension=dimension,
                                                           window_size=window_size, neg_sample_rate=neg_sample_rate,
                                                           batch_size=batch_size, min_count=min_count)
        self.cluster_center_matrix = None
        self.cluster_word_count = None
        self.sense_limit = sense_limit
        self.threshold = threshold
        self.learn_multi_vec = []
        self.distance_type = distance_type
        self.use_dpmeans = use_dpmeans
        self.multi_sense_word_limit = multi_sense_word_limit
        self.single_sense_list = single_sense_list

    def _init_values(self):
        num_multi_sense_words = sum(self.learn_multi_vec)
        total_length = self.words_limit + (self.sense_limit - 1) * num_multi_sense_words
        self.wordvec_matrix = (random.randn(total_length, self.dimension).astype(
            numpy.float32) - 0.5) / self.dimension
        self.weight_matrix = zeros((self.words_limit, self.dimension), dtype=numpy.float32)
        self.biases = zeros(self.words_limit, dtype=numpy.float32)
        self.cluster_center_matrix = zeros((total_length, self.dimension), dtype=numpy.float32)
        self.cluster_word_count = zeros(total_length)

    def dump(self, path):
        if path:
            params = [self.wordvec_matrix,
                      self.weight_matrix,
                      self.biases,
                      self.cluster_center_matrix,
                      self.cluster_word_count,
                      self.word_matrix_index,
                      self.word_list]
            cPickle.dump(params, open(path, 'wb'))

    def load(self, path):
        params = cPickle.load(open(path, 'rb'))
        assert len(params[1]) == self.words_limit, "size dont match, %d vs. %d" % (len(params[1]), self.words_limit)
        self.weight_matrix = params[1]
        self.biases = params[2]
        if len(params) < 7:
            assert len(params[0]) == self.words_limit, "size dont match, %d vs. %d" % (len(params[0]), self.words_limit)
            self.wordvec_matrix[:self.words_limit] = params[0]
        else:
            assert len(params[0]) == len(self.wordvec_matrix), "size dont match, %d vs %d" % \
                                                               (len(params[0]),
                                                                len(self.wordvec_matrix))
            self.wordvec_matrix = params[0]
            self.cluster_center_matrix = params[3]
            self.cluster_word_count = params[4]
            self.word_matrix_index = params[5]
            self.word_list = params[6]

    def _load_words(self, single_sense_list=None, multi_sense_word_limit=None):
        self.words_limit = min(self.words_limit, self.tokenizer.count())
        self.word_list = self.tokenizer.word_list
        for i in range(len(self.word_list)):
            self.word_matrix_index[self.word_list[i]] = [i]
        self.set_learn_multi_vec(single_sense_list, multi_sense_word_limit)
        self._init_values()

    def set_learn_multi_vec(self, single_sense_list, multi_sense_word_limit):
        self.learn_multi_vec = [False] * self.words_limit
        single_sense_list = single_sense_list or []
        count = 0
        for i in xrange(self.words_limit):
            if multi_sense_word_limit is not None and count >= multi_sense_word_limit:
                break
            if self.word_list[i] not in single_sense_list:
                self.learn_multi_vec[i] = True
                count += 1

    def fit(self, texts, nb_epoch=1, sampling=True, monitor=None):
        for e in range(nb_epoch):
            for k, (seq, (couples, labels, seq_indices)) in enumerate(self._sequentialize(texts, sampling)):
                sense_dict = {}
                for i in range(0, len(couples) - self.batch_size, self.batch_size):
                    wi = self.clustering(seq, seq_indices[i:i + self.batch_size], sense_dict)
                    wj = [c[1] for c in couples[i:i + self.batch_size]]
                    self.trainer.update(self.wordvec_matrix, self.weight_matrix, self.biases,
                                        labels[i:i + self.batch_size], wi, wj)

    def clustering(self, seq, seq_indices, sense_dict):
        wi_new = []
        for si in seq_indices:
            if si in sense_dict:
                wi_new.append(sense_dict[si])
            else:
                wi = seq[si]
                if self.learn_multi_vec[wi]:
                    wi_new.append(self.dpmeans(seq, si) if self.use_dpmeans else self.kmeans(seq, si))
                else:
                    wi_new.append(wi)
                sense_dict[si] = wi_new[-1]
        return wi_new

    def find_nearest_sense(self, word, postion):
        d = [dist_func[self.distance_type](self.cluster_center(wi), postion) for wi in self.word_matrix_index[word]]
        mi = numpy.argmin(d)
        return self.word_matrix_index[word][mi], d[mi], var(d)

    def kmeans(self, seq, si):
        # TODO: kmeans initialize
        context_embedding = self.context_embedding(seq, si)
        word = self.word_list[seq[si]]
        sense = self.find_nearest_sense(word, context_embedding)[0]
        self.cluster_center_matrix[sense] += context_embedding
        self.cluster_word_count[sense] += 1
        return sense

    def dpmeans(self, seq, si):
        context_embedding = self.context_embedding(seq, si)
        word = self.word_list[seq[si]]
        sense = seq[si]
        if self.cluster_word_count[sense] > 0:
            sense, min_dis = self.find_nearest_sense(word, context_embedding)[:2]
            if self.sense_count(word) < self.sense_limit and min_dis > self.threshold:
                sense = len(self.word_list)
                self.word_list.append(word)
                self.word_matrix_index[word].append(sense)
        self.cluster_center_matrix[sense] += context_embedding
        self.cluster_word_count[sense] += 1
        return sense

    def context_embedding(self, seq, si):
        return numpy.mean(self.weight_matrix[self.context_words_indices(seq, si)], axis=0)

    def cluster_center(self, sense):
        return self.cluster_center_matrix[sense] / self.cluster_word_count[sense]

    def sense_count(self, word):
        return len(self.word_matrix_index[word])


class InteractiveClSgNsEmbeddingModel(ClusteringSgNsEmbeddingModel):
    def __init__(self, words_limit=5000, dimension=128, window_size=5, neg_sample_rate=1., batch_size=8, sense_limit=5,
                 threshold=.5, min_count=5, distance_type='COS', use_dpmeans=True, ask_threshold=.4,
                 context_words_limit=15):
        super(InteractiveClSgNsEmbeddingModel, self).__init__(words_limit, dimension, window_size, neg_sample_rate,
                                                              batch_size, sense_limit, threshold, min_count,
                                                              distance_type, use_dpmeans)
        self.ask_threshold = ask_threshold
        self.context_words_limit = context_words_limit
        self.context_words_map = collections.defaultdict(set)

    def fit(self, texts, nb_epoch=1, sampling=True, monitor=None):
        batch_size = self.batch_size
        user = UserClassifier()
        for e in range(nb_epoch):
            for k, (seq, (couples, labels, seq_indices)) in enumerate(self._sequentialize(texts, sampling)):
                sense_dict = {}
                for i in range(0, len(couples) - batch_size, batch_size):
                    wi, wj, labels, questions = self.clustering_ask(seq, seq_indices[i:i + batch_size], sense_dict,
                                                                    [c[1] for c in couples[i:i + batch_size]],
                                                                    labels[i:i + batch_size])
                    self.trainer.update(self.wordvec_matrix,
                                        self.weight_matrix,
                                        self.biases,
                                        labels,
                                        wi, wj)
                    # questions = {
                    #     34:{
                    #         'stem': '',
                    #         'options': {
                    #             12:['','']
                    #         },
                    #         'wj': 21,
                    #         'label':1
                    #     }
                    # }
                    answers = user.ask(questions)
                    wi_usr, wj_usr, label_usr = [], [], []
                    for u, v in answers.iteritems():
                        wi_usr.append(v)
                        wj_usr.append(questions[u]['wj'])
                        label_usr.append(questions[u]['label'])
                    self.trainer.update(self.wordvec_matrix,
                                        self.weight_matrix,
                                        self.biases,
                                        label_usr,
                                        wi_usr, wj_usr)

    def clustering_ask(self, seq, seq_indices, sense_dict, wj_org, label_org):
        wi = []
        wj = []
        labels = []
        questions = []
        for si, c, l in zip(seq_indices, wj_org, label_org):
            if sense_dict.get(si) is not None:
                wi.append(sense_dict[si])
                wj.append(c)
                labels.append(l)
            else:
                wi = seq[si]
                if self.learn_multi_vec[wi]:
                    sense, asking = self.dpmeans(seq, si) if self.use_dpmeans else self.kmeans(seq, si)
                    if asking:
                        # TODO: create questions
                        pass
                    else:
                        wi.append(sense)
                        wj.append(c)
                        labels.append(l)
                        sense_dict[si] = sense
                else:
                    wi.append(wi)
                    wj.append(c)
                    labels.append(l)
                    sense_dict[si] = wi
        return wi, wj, labels, questions

    def kmeans(self, seq, si):
        # TODO: kmeans initialize
        context_embedding = self.context_embedding(seq, si)
        word = self.word_list[seq[si]]
        sense, min_dist, dist_var = self.find_nearest_sense(word, context_embedding)
        asking = dist_var < self.ask_threshold
        if not asking:
            self.cluster_center_matrix[sense] += context_embedding
            self.cluster_word_count[sense] += 1
            self.add_sense_context_words(sense, self.context_words_indices(seq, si))
        return sense, asking

    def dpmeans(self, seq, si):
        context_embedding = self.context_embedding(seq, si)
        word = self.word_list[seq[si]]
        sense = seq[si]
        asking = False
        if self.cluster_word_count[sense] > 0:
            sense, min_dis, dist_var = self.find_nearest_sense(word, context_embedding)
            if self.sense_count(word) < self.sense_limit and min_dis > self.threshold:
                sense = len(self.word_list)
                self.word_list.append(word)
                self.word_matrix_index[word].append(sense)
            elif dist_var < self.ask_threshold:
                asking = True
        if not asking:
            self.cluster_center_matrix[sense] += context_embedding
            self.cluster_word_count[sense] += 1
            self.add_sense_context_words(sense, self.context_words_indices(seq, si))
        return sense, asking

    def get_sense_context_words(self, wi):
        return {sense: self.context_words_map[sense] for sense in self.get_senses(wi)}

    def add_sense_context_words(self, sense, word_indices):
        self.context_words_map[sense] |= set(word_indices)
        if len(self.context_words_map[sense]) > self.context_words_limit:
            l = numpy.asarray(list(self.context_words_map[sense]))
            ids = nearest_k_points(self.cluster_center(sense), [self.weight_matrix[c] for c in l],
                                   self.context_words_limit, self.distance_type).keys()
            self.context_words_map[sense].intersection_update(l[ids])
