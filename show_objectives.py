import pickle
import argparse
import matplotlib.pyplot as plt


def show_obj(path):
    if path:
        with open(path, 'rb') as rf:
            obj = pickle.load(rf)
        plt.plot(obj)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', metavar='OBJECTIVE', help='obj trajectory ot load', type=str, required=False)
    args = parser.parse_args()
    show_obj(args.obj)
