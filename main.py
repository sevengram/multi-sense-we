# -*- coding:utf-8 -*-

from models import SkipGramNegSampEmbeddingModel


def text_generator(path):
    f = open(path)
    for line in f:
        yield line
    f.close()


if __name__ == '__main__':
    model = SkipGramNegSampEmbeddingModel(dimension=64)
    model.build_vocab(text_generator("./data/tiny_text8"))
    model.fit(text_generator("./data/tiny_text8"), sampling=False)
    model.save_model("/mnt/my_data/model/my_models")
