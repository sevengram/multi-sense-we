# -*- coding:utf-8 -*-

import cPickle
import collections

import numpy
from numpy import random, zeros, var, finfo
from theano import tensor as T
from scipy.spatial import distance

import sequence
import text
from trainers import SGD, AdaGrad
from user import UserClassifier

MONITOR_GAP = 20
SNAPSHOT_GAP = 1000
MIN_FLOAT = finfo(numpy.float32).eps

dist_func = {
    'COS': distance.cosine,
    'EUC': distance.euclidean
}


def nearest_k_points(target, points, k, dist_type):
    d = [dist_func[dist_type](target, p) for p in points]
    return {i: d[i] for i in numpy.argpartition(d, k)[:k]}


def monitor_obj(monitor, k, obj, switcher):
    if callable(monitor) and switcher:
        monitor(k, obj)


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
        for i in xrange(len(self.word_list)):
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
        result = []
        if self.wordvec_matrix is None:
            return result
        senses = self.word_matrix_index.get(word, [])
        for si in senses:
            if si is not None and si < self.wordvec_matrix.shape[0]:
                result.append({self.word_list[i]: v for i, v in
                               nearest_k_points(self.wordvec_matrix[si], self.wordvec_matrix, limit,
                                                dist_type).iteritems()})
        return result

    def get_senses(self, wi):
        return self.word_matrix_index[self.word_list[wi]]

    def get_words(self, word_indices):
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

    def fit(self, texts, nb_epoch=1, sampling=True, monitor=None, snapshot_path=None):
        for e in xrange(nb_epoch):
            print("\nEpoch %s..." % e)
            for k, (seq, (couples, labels, seq_indices)) in enumerate(self._sequentialize(texts, sampling)):
                c = numpy.array(couples)
                monitor_obj(monitor, k, self.get_obj(c[:, 0], c[:, 1], labels), switcher=(k == 0))
                for i in xrange(0, len(couples), self.batch_size):
                    wi, wj = numpy.array(zip(*couples[i:i + self.batch_size]))
                    self.trainer.update(self.wordvec_matrix, self.weight_matrix, self.biases,
                                        labels[i:i + self.batch_size], wi, wj)
                monitor_obj(monitor, k, self.get_obj(c[:, 0], c[:, 1], labels), switcher=(k % MONITOR_GAP == 0))
            self.take_snapshot(snapshot_path, e)

    def context_words_indices(self, seq, si, with_si=False):
        return seq[max(0, si - self.window_size): si] + ([si] if with_si else []) + seq[(si + 1):si + self.window_size]

    def context_text(self, seq, si):
        return ' '.join(self.get_words(self.context_words_indices(seq, si, True)))

    def get_obj(self, wi, wj, labels):
        return self.trainer.objective_value(self.wordvec_matrix[wi], self.weight_matrix[wj], self.biases[wj], labels)

    def take_snapshot(self, path, name):
        if path:
            self.dump('%s_%s.pkl' % (path, name))


class ClusteringSgNsEmbeddingModel(SkipGramNegSampEmbeddingModel):
    def __init__(self, words_limit=5000, dimension=128, window_size=5, neg_sample_rate=1., batch_size=8, sense_limit=5,
                 threshold=.5, min_count=5, distance_type='COS', use_dpmeans=True, multi_sense_word_limit=None,
                 single_sense_list=None, context_words_limit=15):
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
        self.context_words_limit = context_words_limit
        self.context_words_map = collections.defaultdict(set)

    def _init_values(self):
        num_multi_sense_words = sum(self.learn_multi_vec)
        total_length = self.words_limit + (self.sense_limit - 1) * num_multi_sense_words
        self.wordvec_matrix = (random.randn(total_length, self.dimension).astype(
            numpy.float32) - 0.5) / self.dimension
        self.weight_matrix = zeros((self.words_limit, self.dimension), dtype=numpy.float32) + MIN_FLOAT
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
                      self.word_list,
                      self.context_words_map]
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
            self.context_words_map = params[7]

    def _load_words(self, single_sense_list=None, multi_sense_word_limit=None):
        self.words_limit = min(self.words_limit, self.tokenizer.count())
        self.word_list = self.tokenizer.word_list
        for i in xrange(len(self.word_list)):
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

    def fit(self, texts, nb_epoch=1, sampling=True, monitor=None, snapshot_path=None):
        for e in xrange(nb_epoch):
            print("\nEpoch %s..." % e)
            for k, (seq, (couples, labels, seq_indices)) in enumerate(self._sequentialize(texts, sampling)):
                c = numpy.array(couples)
                monitor_obj(monitor, k, self.get_obj(c[:, 0], c[:, 1], labels), switcher=(k == 0))
                sense_dict = {}
                wi_new = []
                for i in xrange(0, len(couples), self.batch_size):
                    wi = self.clustering(seq, seq_indices[i:i + self.batch_size], sense_dict)
                    wi_new += wi
                    wj = [p[1] for p in couples[i:i + self.batch_size]]
                    self.trainer.update(self.wordvec_matrix, self.weight_matrix, self.biases,
                                        labels[i:i + self.batch_size], wi, wj)
                monitor_obj(monitor, k, self.get_obj(wi_new, c[:, 1], labels), switcher=(k % MONITOR_GAP == 0))
            self.take_snapshot(snapshot_path, e)

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
        self.update_cluster_center(sense, context_embedding)
        self.add_sense_context_words(sense, self.context_words_indices(seq, si))
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
        self.update_cluster_center(sense, context_embedding)
        self.add_sense_context_words(sense, self.context_words_indices(seq, si))
        return sense

    def context_embedding(self, seq, si):
        return numpy.mean(self.weight_matrix[self.context_words_indices(seq, si)], axis=0)

    def cluster_center(self, sense):
        return self.cluster_center_matrix[sense] / self.cluster_word_count[sense]

    def sense_count(self, word):
        return len(self.word_matrix_index[word])

    def update_cluster_center(self, sense, embedding):
        self.cluster_center_matrix[sense] += embedding
        self.cluster_word_count[sense] += 1

    def get_sense_context_words(self, wi):
        return {sense: self.get_words(self.context_words_map[sense]) for sense in self.get_senses(wi)}

    def add_sense_context_words(self, sense, word_indices):
        self.context_words_map[sense] |= set(word_indices)
        if len(self.context_words_map[sense]) > self.context_words_limit:
            l = numpy.asarray(list(self.context_words_map[sense]))
            ids = nearest_k_points(self.cluster_center(sense), [self.weight_matrix[c] for c in l],
                                   self.context_words_limit, self.distance_type).keys()
            self.context_words_map[sense].intersection_update(l[ids])


class InteractiveClSgNsEmbeddingModel(ClusteringSgNsEmbeddingModel):
    def __init__(self, words_limit=5000, dimension=128, window_size=5, neg_sample_rate=1., batch_size=8, sense_limit=5,
                 threshold=.5, min_count=5, distance_type='COS', use_dpmeans=True, ask_threshold=.4,
                 context_words_limit=15, msg_queue=None):
        super(InteractiveClSgNsEmbeddingModel, self).__init__(words_limit, dimension, window_size, neg_sample_rate,
                                                              batch_size, sense_limit, threshold, min_count,
                                                              distance_type, use_dpmeans, context_words_limit)
        self.ask_threshold = ask_threshold
        self.user = UserClassifier(msg_queue)

    def fit(self, texts, nb_epoch=1, sampling=True, monitor=None, snapshot_path=None):
        self.user.create_task()
        print("Task created!")
        for e in xrange(nb_epoch):
            print("\nEpoch %s..." % e)
            for k, (seq, (couples, labels, seq_indices)) in enumerate(self._sequentialize(texts, sampling)):
                sense_dict = {}
                c = numpy.array(couples)
                monitor_obj(monitor, k, self.get_obj(c[:, 0], c[:, 1], labels), switcher=(k == 0))
                wi_new, wj_new, lables_new = [], [], []
                for i in xrange(0, len(couples), self.batch_size):
                    wi, wj, labels, questions = self.clustering_ask(seq, seq_indices[i:i + self.batch_size], sense_dict,
                                                                    [c[1] for c in couples[i:i + self.batch_size]],
                                                                    labels[i:i + self.batch_size])
                    wi_new += wi
                    wj_new += wj
                    lables_new += labels
                    self.trainer.update(self.wordvec_matrix, self.weight_matrix, self.biases,
                                        labels, wi, wj)
                    if questions:
                        answers = self.user.ask(questions)
                        wi_usr, wj_usr, labels_usr = [], [], []
                        for si, sense in answers.iteritems():
                            wi_usr.append(sense)
                            wj_usr.append(questions[si]['wj'])
                            labels_usr.append(questions[si]['label'])
                            self.update_cluster_center(sense, questions[si]['embedding'])
                            self.add_sense_context_words(sense, self.context_words_indices(seq, si))
                        wi_new += wi_usr
                        wj_new += wj_usr
                        lables_new += labels_usr
                        self.trainer.update(self.wordvec_matrix, self.weight_matrix, self.biases,
                                            labels_usr, wi_usr, wj_usr)
                monitor_obj(monitor, k, self.get_obj(wi_new, wj_new, lables_new), switcher=(k % MONITOR_GAP == 0))
            self.take_snapshot(snapshot_path, e)

    def clustering_ask(self, seq, seq_indices, sense_dict, wj, labels):
        wi_new, wj_new, labels_new = [], [], []
        questions = {}
        for si, j, l in zip(seq_indices, wj, labels):
            if si in sense_dict:
                wi_new.append(sense_dict[si])
                wj_new.append(j)
                labels_new.append(l)
            else:
                wi = seq[si]
                if self.learn_multi_vec[wi]:
                    sense, asking, embedding = self.dpmeans(seq, si) if self.use_dpmeans else self.kmeans(seq, si)
                    if asking:
                        questions[si] = {
                            'stem': self.context_text(seq, si),
                            'options': self.get_sense_context_words(wi),
                            'wj': j,
                            'label': l,
                            'embedding': embedding
                        }
                    else:
                        wi_new.append(sense)
                        wj_new.append(j)
                        labels_new.append(l)
                        sense_dict[si] = sense
                else:
                    wi_new.append(wi)
                    wj_new.append(j)
                    labels_new.append(l)
                    sense_dict[si] = wi
        return wi_new, wj_new, labels_new, questions

    def kmeans(self, seq, si):
        # TODO: kmeans initialize
        context_embedding = self.context_embedding(seq, si)
        word = self.word_list[seq[si]]
        sense, min_dist, dist_var = self.find_nearest_sense(word, context_embedding)
        asking = dist_var < self.ask_threshold
        if not asking:
            self.update_cluster_center(sense, context_embedding)
            self.add_sense_context_words(sense, self.context_words_indices(seq, si))
        return sense, asking, context_embedding

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
            elif self.sense_count(word) > 1 and dist_var < self.ask_threshold:
                asking = True
        if not asking:
            self.update_cluster_center(sense, context_embedding)
            self.add_sense_context_words(sense, self.context_words_indices(seq, si))
        return sense, asking, context_embedding