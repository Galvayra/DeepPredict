# -*- coding: utf-8 -*-
import time
import pickle
from DeepPredict.variables import DIR_NAME, FILE_NAME, IS_CLOSED, NUM_FOLDS, show_options
from DeepPredict.learning.method import *

start_time = time.time()
show_options()

if __name__ == '__main__':
    try:
        with open(DIR_NAME + FILE_NAME, 'rb') as file:
            myData = pickle.load(file)
    except FileNotFoundError:
        print("\nPlease execute encoding script !")
    else:
        if IS_CLOSED:
            closed_validation(myData)
        else:
            if NUM_FOLDS > 1:
                k_fold_cross_validation(myData)

        end_time = time.time()
        print("processing time     --- %s seconds ---" % (time.time() - start_time))