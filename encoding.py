# -*- coding: utf-8 -*-
from DeepPredict.dataset.dataHandler import DataHandler
from DeepPredict.variables import DIR_NAME, FILE_NAME
from DeepPredict.modeling.vectorization import MyVector

import pickle
import os


if __name__ == '__main__':
    if not os.path.isdir(DIR_NAME):
        os.mkdir(DIR_NAME)

    with open(DIR_NAME + FILE_NAME, 'wb') as file:
        myData = MyVector(DataHandler())
        print(myData.vector_list)
        # pickle.dump(myData, file)
