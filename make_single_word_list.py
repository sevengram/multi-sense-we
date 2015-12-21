import cPickle
import os.path as op
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--listpkl', help='pkl file to store word list', type=str)
    parser.add_argument('--wordfile', help='word file to add', type=str)
    args = parser.parse_args()

    wordlist = []

    if op.isfile(args.listpkl):
        with open(args.listpkl, 'rb') as rf:
            wordlist = cPickle.load(rf)

    with open(args.wordfile) as f:
        for w in f:
            wordlist.append(w[:-1])

    with open(args.listpkl, 'wb') as wf:
        cPickle.dump(wordlist, wf)
