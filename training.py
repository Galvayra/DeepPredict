# -*- coding: utf-8 -*-
import time
import pickle
from DeepPredict.variables import DIR_NAME, FILE_NAME

start_time = time.time()

if __name__ == '__main__':
    try:
        with open(DIR_NAME + FILE_NAME, 'rb') as file:
            myData = pickle.load(file)
    except FileNotFoundError:
        print("\nPlease execute encoding script !")
    else:
        end_time = time.time()
        print("processing time     --- %s seconds ---" % (time.time() - start_time))