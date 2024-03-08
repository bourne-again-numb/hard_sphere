"""
python utilities
"""

import csv
import numpy as np

PREC = 5


def read_csv(fileame):
    """ read a CSV file """
    raw_data = []
    with open(fileame, 'r') as handle:
        reader_obj = csv.reader(handle)
        for row in reader_obj:
            raw_data.append(row)

    return raw_data


def my_round(obj, precision=PREC):
    """ custom round function """
    if isinstance(obj, tuple):
        result = tuple([round(float(elem), precision) for elem in obj])
    elif isinstance(obj, list):
        result = [round(float(elem), precision) for elem in obj]
    else:
        result = round(float(obj), precision)
    return result


class RandomNumberGenerator:
    """ random number generator """

    def __init__(self, size: int = 1000000):
        """ size of he default randon number geenrator list """
        self._size = size
        self._counter = 0
        self._random_number_list = None

    def initialize(self):
        print("Initializing Random Number")
        self._counter = 0
        self._random_number_list = np.random.rand(self._size)

    @property
    def get(self) -> float:
        """ get a random number """

        if (self._counter == self._size - 1) or (self._random_number_list is None):
            self.initialize()

        result = self._random_number_list[self._counter]
        self._counter += 1
        return result
