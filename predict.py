# -*- coding: utf-8 -*-
from DeepPredict.modeling.variables import DUMP_PATH, DUMP_FILE, IS_CLOSED, NUM_FOLDS, RATIO
from DeepPredict.learning.train import MyTrain
import json


if __name__ == '__main__':
    if IS_CLOSED:
        file_name = DUMP_PATH + DUMP_FILE + "_closed"
    else:
        file_name = DUMP_PATH + DUMP_FILE + "_opened_" + str(NUM_FOLDS)
    try:
        with open(file_name, 'r') as file:
            vector_list = json.load(file)
    except FileNotFoundError:
        print("\nPlease execute encoding script !")
        print("make sure whether vector file is existed in", DUMP_PATH, "directory")
    else:
        if IS_CLOSED:
            print("\n\n========== CLOSED DATA SET ==========\n")
        else:
            print("\n\n========== OPENED DATA SET ==========\n")
            print("k fold -", NUM_FOLDS)
            if NUM_FOLDS == 1:
                print("test ratio -", str(RATIO)+"%")
            print()

        train = MyTrain(vector_list, IS_CLOSED)
        train.training()
