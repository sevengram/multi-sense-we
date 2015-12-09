# -*- coding:utf-8 -*-

import cPickle
import sys

import numpy
from numpy import random, zeros, var
from theano import tensor as T
from scipy.spatial import distance

import sequence
import text
from trainers import SGD, AdaGrad

MONITOR_GAP = 20
SNAPSHOT_GAP = 2000

dist_func = {
    'COS': distance.cosine,
    'EUC': distance.euclidean
}


class WordEmbeddingModel(object):
    def __init__(self, words_limit=5000, dimension=128, min_count=5, use_stop_words=False):
        self.words_limit = words_limit
        self.dimension = dimension
        self.tokenizer = None
        self.word_list = []
        self.word_matrix_index = {}
        self.wordvec_matrix = None
        self.weight_matrix = None
        self.biases = None
        self.min_count = min_count
        self.use_stop_words = use_stop_words

    def _init_values(self):
        self.wordvec_matrix = (random.randn(self.words_limit, self.dimension).astype(
            numpy.float32) - 0.5) / self.dimension
        self.weight_matrix = zeros((self.words_limit, self.dimension), dtype=numpy.float32)
        self.biases = zeros(self.words_limit, dtype=numpy.float32)

    def build_vocab(self, texts, *args, **kwargs):
        self.tokenizer = text.Tokenizer()
        self.tokenizer.fit_on_texts(texts, words_limit=self.words_limit, min_count=self.min_count,
                                    use_stop_words=self.use_stop_words)
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

    def nearest_words(self, word, limit=20, distance_type='COS'):
        if self.wordvec_matrix is None:
            print('load vocab and model first!')
            return None
        word_index = self.word_matrix_index.get(word)[0]
        if word_index is None or word_index >= self.wordvec_matrix.shape[0]:
            print('can\'t find this word!')
            return None
        else:
            d = [dist_func[distance_type](self.wordvec_matrix[word_index], v) for v in self.wordvec_matrix]
            nearest_indices = numpy.argpartition(d, limit)[:limit]
            return {self.word_list[i]: d[i] for i in nearest_indices}


class SkipGramNegSampEmbeddingModel(WordEmbeddingModel):
    def __init__(self, words_limit=5000, dimension=128, window_size=5, neg_sample_rate=1.,
                 batch_size=8, min_count=5, use_stop_words=False):
        super(SkipGramNegSampEmbeddingModel, self).__init__(words_limit=words_limit, dimension=dimension,
                                                            min_count=min_count, use_stop_words=use_stop_words)
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
            if take_snapshot and e is not (nb_epoch-1):
                self.dump(snapshot_path + '_' + str(e) + '.pkl')
            print "\n"


class ClusteringSgNsEmbeddingModel(SkipGramNegSampEmbeddingModel):
    def __init__(self, words_limit=5000, dimension=128, window_size=5, neg_sample_rate=1., batch_size=8,
                 max_senses=5, threshold=.5, min_count=5, use_stop_words=False, distance_type='COS',
                 use_dpmeans=True, multi_sense_word_limit=None, single_sense_list=None):
        super(ClusteringSgNsEmbeddingModel, self).__init__(words_limit=words_limit, dimension=dimension,
                                                           window_size=window_size, neg_sample_rate=neg_sample_rate,
                                                           batch_size=batch_size, min_count=min_count,
                                                           use_stop_words=use_stop_words)
        self.cluster_center_matrix = None
        self.cluster_word_count = None
        self.max_senses = max_senses
        self.threshold = threshold
        self.learn_multi_vec = []
        self.distance_type = distance_type
        self.use_dpmeans = use_dpmeans
        self.multi_sense_word_limit = multi_sense_word_limit
        self.single_sense_list = single_sense_list

    def _init_values(self):
        num_multi_sense_words = sum(self.learn_multi_vec)
        total_length = self.words_limit + (self.max_senses - 1) * num_multi_sense_words
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

    def fit(self, texts, nb_epoch=1, monitor=None, sampling=True, take_snapshot=False, snapshot_path=None):
        batch_size = self.batch_size
        for e in range(nb_epoch):
            print "Epoch", e, "..."
            for k, (seq, (couples, labels, seq_indices)) in enumerate(self._sequentialize(texts, sampling)):
                sense_dict = {}
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
                    wi = self.clustering(seq, seq_indices[i:i + batch_size], sense_dict)
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
            if take_snapshot and e is not (nb_epoch - 1):
                self.dump(snapshot_path + '_' + str(e) + '.pkl')

            print "\n"

    def clustering(self, seq, seq_indices, sense_dict):
        wi_new = []
        for si in seq_indices:
            if sense_dict.get(si) is not None:
                wi_new.append(sense_dict[si])
            else:
                wi = seq[si]
                if self.learn_multi_vec[wi]:
                    wi_new.append(self.dpmeans(seq, si) if self.use_dpmeans else self.kmeans(seq, si))
                else:
                    wi_new.append(wi)
                sense_dict[si] = wi_new[-1]
        return wi_new

    def find_nearest_sense(self, word, context):
        sense, min_dis = 0, sys.maxsize
        for wi in self.word_matrix_index[word]:
            dist = dist_func[self.distance_type](self.cluster_center_matrix[wi] / self.cluster_word_count[wi], context)
            if dist < min_dis:
                sense, min_dis = wi, dist
        return sense, min_dis

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
            sense, min_dis = self.find_nearest_sense(word, context_embedding)
            if self.sense_count(word) < self.max_senses and min_dis > self.threshold:
                sense = len(self.word_list)
                self.word_list.append(word)
                self.word_matrix_index[word].append(sense)
        self.cluster_center_matrix[sense] += context_embedding
        self.cluster_word_count[sense] += 1
        return sense

    def context_embedding(self, seq, si):
        context_words_indices = seq[max(0, si - self.window_size): si] + seq[(si + 1):si + self.window_size]
        return numpy.mean(self.weight_matrix[context_words_indices], axis=0)

    def sense_count(self, word):
        return len(self.word_matrix_index[word])


class InteractiveClSgNsEmbeddingModel(ClusteringSgNsEmbeddingModel):
    def __init__(self, words_limit=5000, dimension=128, window_size=5, neg_sample_rate=1., batch_size=8,
                 max_senses=5, threshold=.5, min_count=5, use_stop_words=False, distance_type='COS',
                 use_dpmeans=True, ask_threshold=.4, show_context=15, max_context=200):
        super(InteractiveClSgNsEmbeddingModel, self).__init__(words_limit, dimension, window_size, neg_sample_rate,
                                                              batch_size, max_senses, threshold, min_count,
                                                              use_stop_words, distance_type, use_dpmeans)
        self.ask_threshold = ask_threshold
        self.show_context = show_context
        self.max_context = max_context
        self.context_list = {}
        self.significant_context_list = {}

    def fit(self, texts, nb_epoch=1, monitor=None, sampling=True, take_snapshot=False, snapshot_path=None):
        batch_size = self.batch_size
        for e in range(nb_epoch):
            print "Epoch", e, "..."
            for k, (seq, (couples, labels, seq_indices)) in enumerate(self._sequentialize(texts, sampling)):
                sense_dict = {}
                if callable(monitor) and k == 0:
                    c = numpy.array(couples)
                    obj = self.trainer.get_objective_value(self.wordvec_matrix[c[:, 0]],
                                                           self.weight_matrix[c[:, 1]],
                                                           self.biases[c[:, 1]],
                                                           labels)
                    monitor(k, obj)

                n = len(couples)
                wi_all = []
                wj_all = []
                label_all = []
                for i in range(0, n - batch_size, batch_size):
                    # get real meaning
                    wj_org = [c[1] for c in couples[i:i + batch_size]]
                    wi, wi_ask, wj, wj_ask, si_ask, l_train, l_ask = \
                        self.clustering_ask(seq, seq_indices[i:i + batch_size], wj_org,
                                            labels[i:i+batch_size], sense_dict)

                    wi_all += wi
                    wj_all += wj
                    label_all += l_train
                    # 可以先训练这些
                    self.trainer.update(self.wordvec_matrix,
                                        self.weight_matrix,
                                        self.biases,
                                        l_train,
                                        wi, wj)

                    wj_ans = []
                    wi_ans = []
                    l_ans = []
                    # 同时你把问题发给server，让用户来判断wi_ask这些单词真实的意思标号。
                    #
                    #                         wj_ask: 词对中的context，需要这个的原因从想要你返回给我的可以明白【暂时先无视这个
                    #                          l_ask: 需要的原因和wj_ask一样
                    #                         wi_ask: 单词的原index，就是tokenizer给出的那个
                    #                         si_ask: 这个单词在当前读入的sequence中的位置
                    #                            seq: 当前的sequence，用来找到某个单词对应的context
                    #                 self.word_list: 从word index或者context index 找到真正的单词
                    #         self.word_matrix_index: 从word找到他现在有的sense list
                    #              self.context_list: 字典类型，key：sense index，value：context list（最多存self.max_context=200 个）；
                    #                                 需要从这里找到距离cluster center最近的self.show_context=15个context index
                    #  self.significant_context_list: 字典类型，key: sense index, value: significant context index (长度为self.show_context=15)
                    #                                 如果某个sense对应的value不是None的话，说明这个意思的significant context list已经固定了，
                    #                                 你就不需要使用 self.context_list了
                    #                     sense_dict: 当参数传进去，更新一下

                    # TODO: get answer from user

                    # 想要你返回给我的:
                    #                         wi_ans: wi_ask中用户选择好的sense
                    #                         si_ans: wi_ans对应的si，这是为了我更新一下self.cluster_centers 和self.cluster_word_count
                    #                  wi_need_merge: wi_ask中用户觉得我们的sense太扯淡需要把他们都merge重新来的
                    #                         wj_ans: wi_ask中用户选择好sense不需要merge的那些单词对应的要训练的context 【所以上一步你需要wj_ask
                    #                          l_ans: 不用merge的词对的label
                    #                     sense_dict: 更新一下sense_dict, key是si, value是用户选择的sense，这样neg sample再遇到就不用再问用户一次了，我后面有判断，只要这里存过的就不问用户

                    self.trainer.update(self.wordvec_matrix,
                                        self.weight_matrix,
                                        self.biases,
                                        l_ans,
                                        wi_ans, wj_ans)

                    ## 完全不知道上述工作可不可以并行。。。还是说只能按照代码的顺序来

                    wi_all += wi_ans     #为了monitor一下objective value也是心累 = =
                    wj_all += wj_ans
                    label_all += l_ans
                if callable(monitor) and k % MONITOR_GAP == 0 and k is not 0:
                    obj = self.trainer.get_objective_value(self.wordvec_matrix[wi_all],
                                                           self.weight_matrix[wj_all],
                                                           self.biases[wj_all],
                                                           label_all)
                    monitor(k, obj)
            if take_snapshot and e is not (nb_epoch - 1):
                self.dump(snapshot_path + '_' + str(e) + '.pkl')

            print "\n"

    def clustering_ask(self, seq, seq_indices, wj_org, label_org,  sense_dict):
        wi_new = []
        wi_new_ask = []
        wj = []
        wj_ask = []
        si_ask = []
        l_train = []
        l_ask = []
        for si, c, l in zip(seq_indices, wj_org, label_org):
            if sense_dict.get(si) is not None:
                wi_new.append(sense_dict[si])
                wj.append(c)
                l_train.append(l)
            else:
                wi = seq[si]
                if self.learn_multi_vec[wi]:
                    if self.use_dpmeans:
                        sense, asking = self.dpmeans(seq, si)
                    else:
                        sense, asking = self.kmeans(seq, si)
                    if asking:
                        wi_new_ask.append(wi)
                        wj_ask.append(c)
                        si_ask.append(si)
                        l_ask.append(l)
                    else:
                        wi_new.append(sense)
                        wj.append(c)
                        l_train.append(l)
                        sense_dict[si] = sense
                else:
                    wi_new.append(wi)
                    wj.append(c)
                    l_train.append(l)
                    sense_dict[si] = wi
        return wi_new, wi_new_ask, wj, wj_ask, si_ask, l_train, l_ask

    def find_nearest_sense(self, word, context):
        sense, min_dis = 0, sys.maxsize
        all_dist = []
        for wi in self.word_matrix_index[word]:
            dist = dist_func[self.distance_type](self.cluster_center_matrix[wi] / self.cluster_word_count[wi], context)
            all_dist.append(dist)
            if dist < min_dis:
                sense, min_dis = wi, dist
        dist_var = var(all_dist)
        return sense, min_dis, dist_var

    def kmeans(self, seq, si):
        # TODO: kmeans initialize
        context_embedding, context_indices = self.context_embedding(seq, si)
        word = self.word_list[seq[si]]
        asking = False
        sense, min_dist, dist_var = self.find_nearest_sense(word, context_embedding)
        if dist_var < self.ask_threshold:
            asking = True
        if not asking:
            self.cluster_center_matrix[sense] += context_embedding
            self.cluster_word_count[sense] += 1
            if self.context_list.get(sense) is None:
                self.context_list[sense] = context_indices
            else:
                if len(self.context_list[sense]) < self.max_context:
                    self.context_list[sense].append(context_indices)
                elif self.significant_context_list.get(sense) is None:
                    self.significant_context_list[sense] = self.find_significant_context(sense)
        return sense, asking

    def dpmeans(self, seq, si):
        context_embedding, context_indices = self.context_embedding(seq, si)
        word = self.word_list[seq[si]]
        sense = seq[si]
        asking = False
        if self.cluster_word_count[sense] > 0:
            sense, min_dis, dist_var = self.find_nearest_sense(word, context_embedding)
            if self.sense_count(word) < self.max_senses and min_dis > self.threshold:
                sense = len(self.word_list)
                self.word_list.append(word)
                self.word_matrix_index[word].append(sense)
            else:
                if dist_var < self.ask_threshold:
                    asking = True
        if not asking:
            self.cluster_center_matrix[sense] += context_embedding
            self.cluster_word_count[sense] += 1
            if self.context_list.get(sense) is None:
                self.context_list[sense] = context_indices
            else:
                if len(self.context_list[sense]) < self.max_context:
                    self.context_list[sense].append(context_indices)
                elif self.significant_context_list.get(sense) is None:
                    self.significant_context_list[sense] = self.find_significant_context(sense)

        return sense, asking

    def find_significant_context(self, sense):
        mu = self.cluster_center_matrix[sense]/self.cluster_word_count
        d = [dist_func[self.distance_type](mu, self.weight_matrix[c]) for c in self.context_list[sense]]
        nearest_indices = numpy.argpartition(d, self.show_context)[:self.show_context]
        sig_context = []
        for i, c in enumerate(self.context_list[sense]):
            if i in nearest_indices:
                sig_context.append(c)
        return sig_context

    def context_embedding(self, seq, si):
        context_words_indices = seq[max(0, si - self.window_size): si] + seq[(si + 1):si + self.window_size]
        return numpy.mean(self.weight_matrix[context_words_indices], axis=0), context_words_indices
