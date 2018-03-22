# -*- coding: utf-8 -*-

import sys
from os import path

try:
    import DeepPredict
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DeepPredict.dataset.dataHandler import DataHandler
from DeepPredict.modeling.vectorization import MyVector


if __name__ == '__main__':
    myData = MyVector(DataHandler())
    myData.encoding()
    myData.dump()

