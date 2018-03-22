# -*- coding: utf-8 -*-

import sys
from os import path

try:
    from DeepPredict.dataset.dataHandler import DataHandler
    from DeepPredict.modeling.vectorization import MyVector
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from DeepPredict.dataset.dataHandler import DataHandler
    from DeepPredict.modeling.vectorization import MyVector

if __name__ == '__main__':
    myData = MyVector(DataHandler())
    myData.encoding()
    myData.dump()

    # for pos, i in enumerate(myData.vector_list):
    #     print(pos + 1, "data")
    #     for k, v in i.items():
    #         print(k, len(v), len(v[0]))
