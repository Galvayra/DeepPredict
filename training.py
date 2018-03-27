# -*- coding: utf-8 -*-
import sys
import json
from os import path

try:
    import DeepPredict
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DeepPredict.dataset.variables import DATA_FILE
from DeepPredict.modeling.variables import DUMP_PATH, DUMP_FILE, TENSOR_PATH
from DeepPredict.learning.train import MyTrain
import DeepPredict.arguments as op


if __name__ == '__main__':
    csv_name = DATA_FILE.split('.')[0]

    if op.USE_W2V:
        append_name = "w2v_"
    else:
        append_name = ""

    if op.IS_CLOSED:
        file_name = append_name + csv_name + "_closed"
    else:
        file_name = append_name + csv_name + "_opened_" + str(op.NUM_FOLDS)
    try:
        with open(DUMP_PATH + DUMP_FILE + "_" + file_name, 'r') as file:
            vector_list = json.load(file)
    except FileNotFoundError:
        print("\nPlease execute encoding script !")
        print("make sure whether vector file is existed in", DUMP_PATH, "directory")
    else:
        print("\nRead vectors -", file_name)
        op.show_options()

        train = MyTrain(vector_list)
        train.training(do_show=op.DO_SHOW, tensor_save=TENSOR_PATH + op.USE_ID + "_" + file_name)
