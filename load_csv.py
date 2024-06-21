import csv

import datasets


def load_csv(path: str) -> datasets.Dataset:
    f = list(csv.reader(open(path)))

    print(f[0], f[1])


if __name__ == '__main__':
    load_csv('training_data/the_conversation.csv')
