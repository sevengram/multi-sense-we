# -*- coding:utf-8 -*-

import cPickle
import sys

import numpy
from numpy import random, zeros, linalg
from theano import tensor as T
from scipy.spatial.distance import cosine, euclidean

import sequence
import text
from trainers import SGD, AdaGrad

MAXI = sys.maxint
# manager = Manager()
MONITOR_GAP = 20
SNAPSHOT_GAP = 2000


class WordEmbeddingModel(object):
    def __init__(self, words_limit=50000, dimension=128, space_factor=1, min_count=5, use_stop_words=False):
        self.words_limit = words_limit
        self.dimension = dimension
        self.space_factor = space_factor
        self.tokenizer = None
        self.word_list = []
        self.word_matrix_index = {}
        self.wordvec_matrix = None
        self.weight_matrix = None
        self.biases = None
        self.min_count = min_count
        self.use_stop_words = use_stop_words

    def _init_values(self):
        factor = self.space_factor
        self.wordvec_matrix = (random.randn(self.words_limit * factor, self.dimension).astype(
            numpy.float32) - 0.5) / self.dimension
        self.weight_matrix = zeros((self.words_limit * factor, self.dimension), dtype=numpy.float32)
        self.biases = zeros(self.words_limit * factor, dtype=numpy.float32)

    def build_vocab(self, texts):
        self.tokenizer = text.Tokenizer(words_limit=self.words_limit, min_count=self.min_count,
                                        use_stop_words=self.use_stop_words)
        self.tokenizer.fit_on_texts(texts)
        self._load_words()

    def load_vocab(self, path):
        self.tokenizer = cPickle.load(open(path, 'rb'))
        self._load_words()

    def _load_words(self):
        self.words_limit = min(self.words_limit, self.tokenizer.count())
        self.word_list = self.tokenizer.provide_word_list()
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
        self.wordvec_matrix = cPickle.load(open(path, 'rb'))

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

    # def nearest_words(self, word, limit=20):
    #     if self.wordvec_matrix is None:
    #         print('load vocab and model first!')
    #         return None
    #     word_index = self.word_matrix_index.get(word)[0]
    #     if word_index is None or word_index >= self.wordvec_matrix.shape[0]:
    #         print('can\'t find this word!')
    #         return None
    #     else:
    #         d = [linalg.norm(self.wordvec_matrix[word_index] - v) for v in self.wordvec_matrix]
    #         nearest_indices = numpy.argpartition(d, limit)[:limit]
    #         return {self.word_list[i]: d[i] for i in nearest_indices}

    def nearest_words(self, word, limit=20):
        if self.wordvec_matrix is None:
            print('load vocab and model first!')
            return None
        word_index = self.word_matrix_index.get(word)[0]
        if word_index is None or word_index >= self.wordvec_matrix.shape[0]:
            print('can\'t find this word!')
            return None
        else:
            d = []
            for idx, v in enumerate(self.wordvec_matrix):
                d.append((cosine(self.wordvec_matrix[word_index], v), idx))

            d.sort(key=lambda x : x[0])
            nearest_indices = d[:limit]
            return {self.word_list[j]: i for i, j in nearest_indices}


class SkipGramNegSampEmbeddingModel(WordEmbeddingModel):
    def __init__(self, words_limit=5000, dimension=128, space_factor=1, window_size=5, neg_sample_rate=1.,
                 batch_size=8, min_count=5, use_stop_words=False):
        super(SkipGramNegSampEmbeddingModel, self).__init__(words_limit=words_limit, dimension=dimension,
                                                            space_factor=space_factor, min_count=min_count,
                                                            use_stop_words=use_stop_words)
        self.window_size = window_size
        self.batch_size = batch_size
        self.neg_sample_rate = neg_sample_rate
        self.trainer = None

    def _sequentialize(self, texts, sampling=True, **kwargs):
        sampling_table = sequence.make_sampling_table(self.words_limit) if sampling else None
        for seq in self.tokenizer.texts_to_sequences_generator(texts):
            yield seq, sequence.skipgrams(seq, self.words_limit, window_size=self.window_size,
                                          negative_samples=self.neg_sample_rate,
                                          sampling_table=sampling_table)

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

    def fit(self, texts, nb_epoch=1, monitor=None, sampling=True, take_snapshot=False, snapshot_path=None):
        for e in range(nb_epoch):
            print "Epoch", e, "..."
            for k, (seq, (couples, labels, seq_indices)) in enumerate(self._sequentialize(texts, sampling)):
                if callable(monitor) and k == 0:
                    c = numpy.array(couples)
                    obj = self.trainer.get_objective_value(self.wordvec_matrix[c[:, 0]],
                                                           self.weight_matrix[c[:, 1]],
                                                           self.biases[c[:, 1]],
                                                           labels)
                    monitor(k, obj)
                n = len(couples)
                for i in range(0, n - self.batch_size, self.batch_size):
                    wi, wj = numpy.array(zip(*couples[i:i + self.batch_size]))
                    self.trainer.update(self.wordvec_matrix,
                                        self.weight_matrix,
                                        self.biases,
                                        labels[i:i + self.batch_size],
                                        wi, wj)
                if callable(monitor) and k % MONITOR_GAP == 0 and k is not 0:
                    c = numpy.array(couples)
                    obj = self.trainer.get_objective_value(self.wordvec_matrix[c[:, 0]],
                                                           self.weight_matrix[c[:, 1]],
                                                           self.biases[c[:, 1]],
                                                           labels)
                    monitor(k, obj)
                if take_snapshot and k % SNAPSHOT_GAP == 0 and k is not 0:
                    self.dump(snapshot_path + '_' + str(k) + '.pkl')
            print "\n"


class ClusteringSgNsEmbeddingModel(SkipGramNegSampEmbeddingModel):
    def __init__(self, words_limit=5000, dimension=128, space_factor=4, window_size=5, neg_sample_rate=1., batch_size=8,
                 max_senses=5, threshold=1., min_count=5, use_stop_words=False, learn_top_multi=None, skip_list=None,
                 distance_type='COS', use_dpmeans=True):
        super(ClusteringSgNsEmbeddingModel, self).__init__(words_limit=words_limit, dimension=dimension,
                                                           space_factor=space_factor, window_size=window_size,
                                                           neg_sample_rate=neg_sample_rate, batch_size=batch_size,
                                                           min_count=min_count, use_stop_words=use_stop_words)
        self.cluster_center_matrix = None
        self.word_count_inCluster = None
        self.max_senses = max_senses
        self.threshold = threshold
        self.learn_top_multi = learn_top_multi
        self.skip_list = skip_list
        self.learnMultiVec = []
        self.sense_word_list = []
        self.distance_type = distance_type
        self.num_active_we = 0
        self.use_dpmeans = use_dpmeans
        self.cluster_center_matrix = zeros((self.words_limit * self.space_factor, self.dimension), dtype=numpy.float32)
        self.word_count_inCluster = zeros(self.words_limit * self.space_factor, dtype=numpy.float32)

    def dump(self, path):
        if path:
            params = [self.wordvec_matrix,
                      self.weight_matrix,
                      self.biases,
                      self.cluster_center_matrix,
                      self.word_count_inCluster,
                      self.word_matrix_index,
                      self.sense_word_list]
            cPickle.dump(params, open(path, 'wb'))

    def load(self, path):
        params = cPickle.load(open(path, 'rb'))
        assert len(params[1]) == self.words_limit, "size dont match, %d vs. %d" % (len(params[1]), self.words_limit)
        self.weight_matrix = params[1]
        self.biases = params[2]
        if len(params) == 5:
            assert len(params[0]) == self.words_limit, "size dont match, %d vs. %d" % (len(params[0]), self.words_limit)
            self.wordvec_matrix[:self.words_limit] = params[0]
        else:
            assert len(params[0]) == self.words_limit*self.space_factor, "size dont match, %d vs %d" % \
                                                                         (len(params[0]),
                                                                          self.words_limit*self.space_factor)
            self.wordvec_matrix = params[0]
            self.cluster_center_matrix = params[3]
            self.word_count_inCluster = params[4]
            self.word_matrix_index = params[5]
            self.sense_word_list = params[6]

    def build_vocab(self, texts):
        super(ClusteringSgNsEmbeddingModel, self).build_vocab(texts)
        self.sense_word_list = self.word_list
        self.set_learn_vector()

    def load_vocab(self, path):
        super(ClusteringSgNsEmbeddingModel, self).load_vocab(path)
        self.sense_word_list = self.word_list
        self.set_learn_vector()

    def set_learn_vector(self):
        self.num_active_we = self.words_limit
        length = len(self.word_list)
        self.learnMultiVec = [True]*length
        if self.skip_list is not None:
            for i in xrange(length):
                if self.word_list[i] in self.skip_list:
                    self.learnMultiVec[i] = False
        if self.learn_top_multi is not None and self.learn_top_multi < sum(self.learnMultiVec):
            count = 0
            idx = 0
            while count < self.learn_top_multi - 1:
                if self.learnMultiVec[idx]:
                    self.learnMultiVec[idx] = False
                    count += 1
                idx += 1

    def fit(self, texts, nb_epoch=1, monitor=None, sampling=True, take_snapshot=False, snapshot_path=None):
        batch_size = self.batch_size
        for e in range(nb_epoch):
            print "Epoch", e, "..."
            for k, (seq, (couples, labels, seq_indices)) in enumerate(self._sequentialize(texts, sampling)):
                if callable(monitor) and k == 0:
                    c = numpy.array(couples)
                    obj = self.trainer.get_objective_value(self.wordvec_matrix[c[:, 0]],
                                                           self.weight_matrix[c[:, 1]],
                                                           self.biases[c[:, 1]],
                                                           labels)
                    monitor(k, obj)

                n = len(couples)
                wi_all = []
                last = n
                for i in range(0, n - batch_size, batch_size):
                    # get real meaning
                    wi = self.clustering(seq, seq_indices[i:i + batch_size])
                    wj = [c[1] for c in couples[i:i + batch_size]]
                    wi_all += wi

                    self.trainer.update(self.wordvec_matrix,
                                        self.weight_matrix,
                                        self.biases,
                                        labels[i:i + batch_size],
                                        wi, wj)
                    last = i + batch_size
                if callable(monitor) and k % MONITOR_GAP == 0 and k is not 0:
                    c = numpy.array(couples)
                    wj = c[:last, 1]
                    obj = self.trainer.get_objective_value(self.wordvec_matrix[wi_all],
                                                           self.weight_matrix[wj],
                                                           self.biases[wj],
                                                           labels[:last])
                    monitor(k, obj)
                if take_snapshot and k % SNAPSHOT_GAP == 0 and k is not 0:
                    self.dump(snapshot_path + '_' + str(k) + '.pkl')

            print "\n"

    def clustering(self, seq, seq_indices):
        wi_new = []
        for si in seq_indices:
            wi = seq[si]
            if self.learnMultiVec[wi]:
                wi_new.append(self.dpmeans(seq, si) if self.use_dpmeans else self.kmeans(seq, si))
            else:
                wi_new.append(wi)
        return wi_new

    def find_sense_seq(self, seq, seq_indices):
        wi_new = []
        for si in seq_indices:
            wi = seq[si]
            if self.learnMultiVec[wi]:
                wi_new.append(self.find_sense(seq, si))
            else:
                wi_new.append(wi)
        return wi_new

    def kmeans(self, seq, si):
        context_embedding = self.context_embedding(seq, si)
        word = self.sense_word_list[seq[si]]
        current_centers_idx = self.word_matrix_index[word]
        min_dis = MAXI
        sense = 0
        for cc in current_centers_idx:
            if self.distance_type == 'COS':
                dist = cosine(self.cluster_center_matrix[cc]/self.word_count_inCluster[cc], context_embedding)
            elif self.distance_type == 'EUC':
                dist = euclidean(self.cluster_center_matrix[cc]/self.word_count_inCluster[cc], context_embedding)
            if dist < min_dis:
                min_dis = dist
                sense = cc
        self.cluster_center_matrix[sense] += context_embedding
        self.word_count_inCluster[sense] += 1
        return sense

    def find_sense(self, seq, si):
        context_embedding = self.context_embedding(seq, si)
        word = self.sense_word_list[seq[si]]
        current_centers_idx = self.word_matrix_index[word]
        min_dis = MAXI
        sense = 0
        if self.word_count_inCluster[current_centers_idx[0]] < 0.5:
            sense = current_centers_idx[0]
        else:
            for cc in current_centers_idx:
                if self.distance_type == 'COS':
                    dist = cosine(self.cluster_center_matrix[cc]/self.word_count_inCluster[cc], context_embedding)
                elif self.distance_type == 'EUC':
                    dist = euclidean(self.cluster_center_matrix[cc]/self.word_count_inCluster[cc], context_embedding)
                else:
                    NotImplementedError()
                if dist < min_dis:
                    min_dis = dist
                    sense = cc
        return sense

    def dpmeans(self, seq, si):
        context_embedding = self.context_embedding(seq, si)
        word = self.sense_word_list[seq[si]]
        current_centers_idx = self.word_matrix_index[word]
        min_dis = MAXI
        sense = 0
        if self.word_count_inCluster[current_centers_idx[0]] < 0.5:
            sense = current_centers_idx[0]
        else:
            for cc in current_centers_idx:
                if self.distance_type == 'COS':
                    dist = cosine(self.cluster_center_matrix[cc]/self.word_count_inCluster[cc], context_embedding)
                elif self.distance_type == 'EUC':
                    dist = euclidean(self.cluster_center_matrix[cc]/self.word_count_inCluster[cc], context_embedding)
                else:
                    NotImplementedError()
                if dist < min_dis:
                    min_dis = dist
                    sense = cc

            if len(self.word_matrix_index[word]) < self.max_senses:
                if min_dis > self.threshold:
                    sense = self.num_active_we
                    self.num_active_we += 1
                    self.sense_word_list.append(word)
                    self.word_matrix_index[word].append(sense)
                    self.wordvec_matrix[sense] = self.wordvec_matrix[seq[si]]

        self.cluster_center_matrix[sense] += context_embedding
        self.word_count_inCluster[sense] += 1
        return sense

    def context_embedding(self, seq, si):
        context_words_indices = seq[max(0, si - self.window_size): si] + seq[(si+1):si + self.window_size]
        return numpy.mean(self.weight_matrix[context_words_indices], axis=0)


# class ClusteringSgMultiEmbeddingModelMP(ClusteringSgNsEmbeddingModel):
#     def __init__(self, words_limit=5000, dimension=128, space_factor=4, window_size=5, neg_sample_rate=1., batch_size=8,
#                  max_senses=5, threshold=1., min_count=5, use_stop_words=False, learn_top_multi=None, skip_list=None,
#                  distance_type='COS', use_dpmeans=True, num_process=2):
#         super(ClusteringSgMultiEmbeddingModelMP, self).__init__(words_limit=words_limit, dimension=dimension,
#                                                                 space_factor=space_factor, window_size=window_size,
#                                                                 neg_sample_rate=neg_sample_rate, batch_size=batch_size,
#                                                                 max_senses=max_senses, threshold=threshold,
#                                                                 min_count=min_count, use_stop_words=use_stop_words,
#                                                                 learn_top_multi=learn_top_multi, skip_list=skip_list,
#                                                                 distance_type=distance_type, use_dpmeans=use_dpmeans)
#         self.num_process = num_process
#
#     def clustering(self, seq, w_org, seq_indices):
#         l = len(seq_indices)
#         cluster_centers = manager.list(self.cluster_center_matrix)
#         weight_matrix = manager.list(self.weight_matrix)
#         word_count_inCluster = manager.list(self.word_count_inCluster)
#         word_matrix_index = manager.list([self.word_matrix_index])
#         sense_word_list = manager.list(self.sense_word_list)
#         seq_shared = manager.list(seq)
#         seq_indices_shared = manager.list(seq_indices)
#         wsize = manager.list([self.window_size])
#         dist_type = manager.list([self.distance_type])
#         learnable = manager.list(self.learnMultiVec)
#         max_senses = manager.list([self.max_senses])
#         threshold = manager.list([self.threshold])
#         pool = mp.Pool(processes=self.num_process)
#         if self.use_dpmeans:
#             f = partial(self.dpmeans_p, learnable, weight_matrix, cluster_centers, word_count_inCluster,
#                         word_matrix_index, sense_word_list, seq_shared, seq_indices_shared, wsize, dist_type, max_senses,
#                         threshold)
#             results = pool.map(f, range(l))
#
#             current_length = self.num_active_we
#             for r, wo in zip(results, w_org):
#                 sense = r[0]
#                 if sense == current_length:
#                     word = self.sense_word_list[wo]
#                     sense = self.num_active_we
#                     self.sense_word_list.append(word)
#                     self.word_matrix_index[word].append(sense)
#                     self.wordvec_matrix[sense] = self.wordvec_matrix[wo]
#                     self.num_active_we += 1
#                 if r[1] is not None:
#                     self.cluster_center_matrix[sense] += r[1]
#                     self.word_count_inCluster[sense] += 1
#
#         else:
#             f = partial(self.kmeans_p, learnable, weight_matrix, cluster_centers, word_count_inCluster,
#                         word_matrix_index, sense_word_list, seq_shared, seq_indices_shared, wsize, dist_type)
#             results = pool.map(f, range(l))
#
#             for r, wo in zip(results, w_org):
#                 sense = r[0]
#                 if r[1] is not None:
#                     self.cluster_center_matrix[sense] += r[1]
#                     self.word_count_inCluster[sense] += 1
#
#         w_new = [r[0] for r in results]
#         pool.close()
#         pool.join()
#         return w_new
#
#     def kmeans_p(self, learnable, weight_matrix, cluster_centers, word_count_inCluster, word_matrix_index, sense_word_list,
#                  seq_shared, seq_indices_shared, wsize, dist_type, i):
#         seq_idx = seq_indices_shared[i]
#         w_idx = seq_shared[seq_idx]
#         if learnable[w_idx]:
#             context_words_indices = seq_shared[max(0, seq_idx - wsize[0]): seq_idx] + \
#                                     seq_shared[(seq_idx+1):seq_idx + wsize[0]]
#             context_embedding = numpy.mean(weight_matrix[context_words_indices], axis=0)
#             current_centers_idx = word_matrix_index[0][sense_word_list[w_idx]]
#             min_dis = MAXI
#             sense = 0
#             for cc in current_centers_idx:
#                 if dist_type[0] == 'COS':
#                     dist = cosine(cluster_centers[cc]/word_count_inCluster[cc], context_embedding)
#                 elif dist_type[0] == 'EUC':
#                     dist = euclidean(cluster_centers[cc]/word_count_inCluster[cc], context_embedding)
#                 if dist < min_dis:
#                     min_dis = dist
#                     sense = cc
#         else:
#             sense = w_idx
#             context_embedding = None
#         return sense, context_embedding
#
#     def dpmeans_p(self, learnable, weight_matrix, cluster_centers, word_count_inCluster, word_matrix_index, sense_word_list,
#                   seq_shared, seq_indices_shared, wsize, dist_type, max_senses, threshold, i):
#         seq_idx = seq_indices_shared[i]
#         w_idx = seq_shared[seq_idx]
#         if learnable[w_idx]:
#             context_words_indices = seq_shared[max(0, seq_idx - wsize[0]): seq_idx] + \
#                                     seq_shared[(seq_idx+1):seq_idx + wsize[0]]
#             context_embedding = weight_matrix[context_words_indices[0]]
#             for cc in context_words_indices[1:]:
#                 context_embedding += weight_matrix[cc]
#             context_embedding /= len(context_words_indices)
#
#             word = sense_word_list[w_idx]
#             current_centers_idx = word_matrix_index[0][word]
#             min_dis = MAXI
#             sense = 0
#
#             if word_count_inCluster[current_centers_idx[0]] < 0.5:
#                 sense = current_centers_idx[0]
#             else:
#                 for cc in current_centers_idx:
#                     if dist_type[0] == 'COS':
#                         dist = cosine(cluster_centers[cc]/word_count_inCluster[cc], context_embedding)
#                     elif dist_type[0] == 'EUC':
#                         dist = euclidean(cluster_centers[cc]/word_count_inCluster[cc], context_embedding)
#                     if dist < min_dis:
#                         min_dis = dist
#                         sense = cc
#
#                 if len(word_matrix_index[0][word]) < max_senses[0]:
#                     if min_dis > threshold[0]:
#                         sense = len(sense_word_list)
#         else:
#             sense = w_idx
#             context_embedding = None
#         return sense, context_embedding
